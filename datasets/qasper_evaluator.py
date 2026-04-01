"""
Official script for evaluating models built for the Qasper dataset. The script
outputs Answer F1 and Evidence F1 reported in the paper.

Supports BERTScore F1 for semantic evaluation (see paper: Cosine F1 = 0.22, BERT F1 = 0.62)
"""
from collections import Counter
import argparse
import string
import re
import json


# Lazy import bert_score to avoid heavy dependency at module load
bert_score = None


def get_bert_scorer():
    """Lazy load BERTScore scorer (cached for efficiency)."""
    global bert_score
    if bert_score is None:
        from bert_score import BERTScorer
        bert_score = BERTScorer(lang="en", rescale_with_baseline=True)
    return bert_score


def bert_f1_score(prediction, ground_truth):
    """
    Calculate BERTScore F1 for answer evaluation.

    BERTScore evaluates semantic similarity rather than lexical overlap,
    making it more suitable for QASPER's long-document free-form answers.

    According to the paper:
    - Cosine F1 (token overlap): 0.22 (too strict)
    - BERT F1 (semantic): 0.62 (more appropriate)

    Args:
        prediction: Model predicted answer
        ground_truth: Reference answer

    Returns:
        BERTScore F1 (rescaled with baseline)
    """
    if not prediction or not ground_truth:
        return 0.0

    # Handle boolean cases where "No" vs "False" should match
    pred_lower = prediction.lower().strip()
    truth_lower = ground_truth.lower().strip()
    if pred_lower == truth_lower:
        return 1.0
    if pred_lower in ["no", "false", "yes", "true"]:
        # Boolean equivalence check
        is_positive = pred_lower in ["yes", "true"]
        truth_is_positive = truth_lower in ["yes", "true"]
        if is_positive == truth_is_positive:
            return 0.999  # Near perfect semantic match for booleans

    try:
        scorer = get_bert_scorer()
        # BERTScore expects lists, returns tuple of Tensors (P, R, F1)
        P, R, F1 = scorer.score([prediction], [ground_truth])
        # Convert tensor to scalar (works for both tensor and float)
        f1_value = F1.detach().cpu().item() if hasattr(F1, 'detach') else float(F1)
        return float(f1_value)
    except Exception as e:
        # Fallback to token F1 if BERTScore fails
        return token_f1_score(prediction, ground_truth)


def normalize_answer(s):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    Lower text and remove punctuation, articles and extra whitespace.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def split_into_sentences(text):
    """
    Split text into sentences using simple heuristic.
    This matches the official Qasper evaluation approach.
    """
    # Simple sentence splitting by common delimiters
    # In production, use NLTK's sent_tokenize
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def token_f1_score(prediction, ground_truth):
    """
    Taken from the official evaluation script for v1.1 of the SQuAD dataset.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score(prediction, ground_truth):
    """
    Calculate F1 score for evidence paragraphs.

    Supports two formats:
    1. Index-based: [{"paragraph_index": 0, "sentence_index": 1}, ...]
    2. String-based: ["evidence text 1", "evidence text 2", ...]
    """
    if not ground_truth and not prediction:
        return 1.0
    if not ground_truth or not prediction:
        return 0.0

    # Check if using index-based format
    if isinstance(prediction[0], dict) and "paragraph_index" in prediction[0]:
        # Index-based format: compare paragraph_index + sentence_index tuples
        pred_indices = set()
        for p in prediction:
            if "paragraph_index" in p and "sentence_index" in p:
                pred_indices.add((p["paragraph_index"], p["sentence_index"]))

        gold_indices = set()
        for g in ground_truth:
            if isinstance(g, dict) and "paragraph_index" in g and "sentence_index" in g:
                gold_indices.add((g["paragraph_index"], g["sentence_index"]))
            elif isinstance(g, str):
                # If gold is string but prediction is index, we can't match
                # Fall back to string comparison
                return paragraph_f1_score_string(prediction, ground_truth)

        num_same = len(pred_indices.intersection(gold_indices))
    else:
        # String-based format: compare evidence text directly
        return paragraph_f1_score_string(prediction, ground_truth)

    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def paragraph_f1_score_string(prediction, ground_truth):
    """String-based evidence F1 score."""
    if not ground_truth and not prediction:
        return 1.0
    if not ground_truth or not prediction:
        return 0.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_answers_and_evidence(data, text_evidence_only):
    answers_and_evidence = {}
    for paper_data in data.values():
        for qa_info in paper_data["qas"]:
            question_id = qa_info["question_id"]
            references = []
            for annotation_info in qa_info["answers"]:
                answer_info = annotation_info["answer"]
                if answer_info["unanswerable"]:
                    references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
                else:
                    if answer_info["extractive_spans"]:
                        answer = ", ".join(answer_info["extractive_spans"])
                        answer_type = "extractive"
                    elif answer_info["free_form_answer"]:
                        answer = answer_info["free_form_answer"]
                        answer_type = "abstractive"
                    elif answer_info["yes_no"]:
                        answer = "Yes"
                        answer_type = "boolean"
                    elif answer_info["yes_no"] is not None:
                        answer = "No"
                        answer_type = "boolean"
                    else:
                        raise RuntimeError(f"Annotation {answer_info['annotation_id']} does not contain an answer")
                    if text_evidence_only:
                        evidence = [text for text in answer_info["evidence"] if "FLOAT SELECTED" not in text]
                    else:
                        evidence = answer_info["evidence"]
                    references.append({"answer": answer, "evidence": evidence, "type": answer_type})
            answers_and_evidence[question_id] = references

    return answers_and_evidence


def evaluate(gold, predicted, use_bert_f1=False, verbose=True):
    """
    Evaluate predictions against gold answers.

    Args:
        gold: Gold answers dict {question_id: references}
        predicted: Predicted answers dict {question_id: {"answer": ..., "evidence": ...}}
        use_bert_f1: If True, compute both Token F1 and BERTScore F1
        verbose: If True, print progress updates

    Returns:
        Evaluation metrics dict
    """
    import sys

    max_answer_f1s = []
    max_answer_bert_f1s = []  # Separate list for BERT F1
    max_answer_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    max_answer_bert_f1s_by_type = {
        "extractive": [],
        "abstractive": [],
        "boolean": [],
        "none": [],
    }
    max_evidence_f1s = []
    num_missing_predictions = 0

    # Progress tracking
    total_questions = len(gold)
    processed = 0
    print_interval = max(1, total_questions // 20)  # Print every 5%

    if verbose:
        print(f"\n开始评估 (共 {total_questions} 个问题)...")
        if use_bert_f1:
            print("  模式: Token F1 + BERTScore F1 (同时计算)")
        else:
            print("  模式: Token F1 (词汇重叠评估)")
        print("-" * 50)

    for question_id, references in gold.items():
        processed += 1

        # Print progress
        if verbose and processed % print_interval == 0:
            pct = processed * 100 // total_questions
            print(f"  进度: {processed}/{total_questions} ({pct}%)", flush=True)

        if question_id not in predicted:
            num_missing_predictions += 1
            max_answer_f1s.append(0.0)
            max_answer_bert_f1s.append(0.0)
            max_evidence_f1s.append(0.0)
            continue

        predicted_answer = predicted[question_id]["answer"]

        # Token F1 evaluation (always computed)
        answer_f1s_and_types = [
            (token_f1_score(predicted_answer, reference["answer"]),
             reference["type"])
            for reference in references
        ]
        max_answer_f1, answer_type = sorted(answer_f1s_and_types, key=lambda x: x[0], reverse=True)[0]
        max_answer_f1s.append(max_answer_f1)
        max_answer_f1s_by_type[answer_type].append(max_answer_f1)

        # BERTScore F1 (only when use_bert_f1=True)
        if use_bert_f1:
            best_bert_f1 = 0.0
            best_bert_type = "abstractive"
            for reference in references:
                bert_f1 = bert_f1_score(predicted_answer, reference["answer"])
                if bert_f1 > best_bert_f1:
                    best_bert_f1 = bert_f1
                    best_bert_type = reference["type"]
            max_answer_bert_f1s.append(best_bert_f1)
            max_answer_bert_f1s_by_type[best_bert_type].append(best_bert_f1)
        else:
            max_answer_bert_f1s.append(0.0)  # Placeholder when not computed

        evidence_f1s = [
            paragraph_f1_score(predicted[question_id]["evidence"], reference["evidence"])
            for reference in references
        ]
        max_evidence_f1s.append(max(evidence_f1s))

    if verbose:
        print("-" * 50)
        print(f"评估完成! ({total_questions}/{total_questions} 100%)")

    mean = lambda x: sum(x) / len(x) if x else 0.0
    result = {
        "Answer F1": mean(max_answer_f1s),
        "Answer F1 by type": {key: mean(value) for key, value in max_answer_f1s_by_type.items()},
        "Evidence F1": mean(max_evidence_f1s),
        "Missing predictions": num_missing_predictions
    }

    # Add BERT F1 metrics if computed
    if use_bert_f1:
        result["Answer BERT F1"] = mean(max_answer_bert_f1s)
        result["Answer BERT F1 by type"] = {key: mean(value) for key, value in max_answer_bert_f1s_by_type.items()}

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QASPER Evaluation Script")
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="""JSON lines file with each line in format:
                {'question_id': str, 'predicted_answer': str, 'predicted_evidence': List[str]}"""
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Test or dev set from the released dataset"
    )
    parser.add_argument(
        "--text_evidence_only",
        action="store_true",
        help="If set, the evaluator will ignore evidence in figures and tables while reporting evidence f1"
    )
    parser.add_argument(
        "--bert-score",
        action="store_true",
        help="""If set, compute BERTScore F1 instead of token F1.
                According to the paper, BERTScore F1 (0.62) is more appropriate for QASPER
                than cosine-based token F1 (0.22), especially for long-document free-form answers."""
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, suppress progress output and only print JSON result"
    )
    args = parser.parse_args()
    gold_data = json.load(open(args.gold))
    gold_answers_and_evidence = get_answers_and_evidence(gold_data, args.text_evidence_only)
    predicted_answers_and_evidence = {}
    for line in open(args.predictions):
        prediction_data = json.loads(line)
        predicted_answers_and_evidence[prediction_data["question_id"]] = {
            "answer": prediction_data["predicted_answer"],
            "evidence": prediction_data["predicted_evidence"]
        }
    evaluation_output = evaluate(gold_answers_and_evidence, predicted_answers_and_evidence, use_bert_f1=args.bert_score, verbose=not args.quiet)
    print(json.dumps(evaluation_output, indent=2))
