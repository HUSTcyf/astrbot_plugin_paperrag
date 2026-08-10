# -*- coding: utf-8 -*-
"""RAGTruth 官方协议幻觉评测：judge 判定"回答是否含幻觉"，对比各模型

协议对齐 RAGTruth 官方评估口径（论文）：response-level（回答级）判定最可靠——
官方报告 span 级 F1 仅 39.5%（GPT-4 14.2%），而回答级 F1 达 86.8%。
因此**指标只算回答级**：judge 对每条回答给出二元判定 hallucinated（是否含幻觉），
span 区间仅作为示例展示的证据，不进入任何指标。

指标：
  hallucination_rate   含幻觉回答数 / 打标成功回答数（主指标，官方口径）
  refusal_rate         拒答占比（官方 prompt 允许 "Unable to answer based on given passages."）

不依赖 ragas：裸 OpenAI 兼容 API + 超时/重试 + 行级磁盘缓存（中断可续跑）。

用法：
  python score_official.py --files answers_base.jsonl answers_sft200.jsonl \
      --out eval_report_official.md
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

CMD_CONFIG = "/mnt/d/AstrBot/data/cmd_config.json"
PROVIDER = "minimax-token-plan"
MODEL = "MiniMax-M3"
MINIMAX_OPENAI_BASE = "https://api.minimaxi.com/v1"
REFUSAL_PHRASE = "unable to answer based on given passages"

JUDGE_SYSTEM = (
    "You are an expert annotator for hallucination detection in Retrieval-Augmented "
    "Generation. Given the retrieved passages, the question and a model-generated answer, "
    "decide whether the answer contains any hallucination.\n"
    "A hallucination is a text segment in the answer that is unsupported by the passages:\n"
    "- Evident Baseless Info: presents information not present in the passages.\n"
    "- Contradictory Info: contradicts the content of the passages.\n"
    "- Irrelevant Info: does not answer the question and is not grounded in the passages.\n"
    "Rules:\n"
    "- Information fully supported by the passages is NOT hallucination.\n"
    '- If the answer refuses to answer (e.g. "Unable to answer based on given passages."), '
    'set "hallucinated" to false.\n'
    '- "spans" is optional supporting evidence only: report the hallucinated spans with '
    "0-based character start and end offsets (end exclusive) and label_type; if there is "
    "no hallucination, leave it an empty list.\n"
    'Output strictly JSON: {"hallucinated": true, "spans": [{"start": 0, "end": 5, '
    '"label_type": "..."}]}'
)

# 无上下文（RAG 检索失败）模式：以参考回答为事实基准判幻觉（truthfulness 口径）
NOCONTEXT_SYSTEM = (
    "You are an expert annotator for hallucination detection in open-domain question "
    "answering without retrieved context. Given the question, a reference answer and a "
    "model-generated answer, decide whether the model answer contains any hallucination.\n"
    "A hallucination is a text segment in the model answer that is factually incorrect or "
    "fabricated, i.e. contradicts the reference answer or introduces unsupported facts.\n"
    "Rules:\n"
    "- Content consistent with the reference answer is NOT hallucination.\n"
    "- Partial answers are allowed; only judge what the model actually claims.\n"
    '- If the answer refuses to answer (e.g. "Unable to answer..."), set "hallucinated" '
    "to false.\n"
    '- "spans" is optional supporting evidence only; leave it an empty list if there is '
    "no hallucination.\n"
    'Output strictly JSON: {"hallucinated": true, "spans": [{"start": 0, "end": 5, '
    '"label_type": "..."}]}'
)


def load_key():
    cfg = json.load(open(CMD_CONFIG, encoding="utf-8-sig"))
    for s in cfg.get("provider_sources", []):
        if s.get("provider") == PROVIDER:
            return s["key"][0] if isinstance(s["key"], list) else s["key"]
    raise SystemExit(f"provider {PROVIDER} not found in {CMD_CONFIG}")


def extract_json_block(text):
    """找第一个平衡的 {...} 块（内容里可能含花括号，不能用贪婪正则）"""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_verdict(content, answer):
    """解析 judge 输出：判定字段必须有效；spans 仅作证据，非法区间丢弃"""
    block = extract_json_block(content)
    if block is None:
        return None
    try:
        data = json.loads(block)
    except (json.JSONDecodeError, AttributeError):
        return None
    hall = data.get("hallucinated")
    if not isinstance(hall, bool):
        return None
    spans = []
    raw = data.get("spans")
    if isinstance(raw, list):
        n = len(answer)
        seen = set()
        for s in raw:
            if not isinstance(s, dict):
                continue
            st, en = s.get("start"), s.get("end")
            if not (isinstance(st, int) and isinstance(en, int)):
                continue
            if not (0 <= st < en <= n) or (st, en) in seen:
                continue
            seen.add((st, en))
            spans.append({"start": st, "end": en, "text": answer[st:en],
                          "label_type": s.get("label_type", "unknown")})
    return {"hallucinated": hall, "spans": spans}


def is_refusal(answer):
    return REFUSAL_PHRASE in answer.lower()


def judge_one(client, question, passages, answer, max_retries, timeout,
              mode="faithful", reference=None):
    if mode == "nocontext":
        system = NOCONTEXT_SYSTEM
        user = (f"Question:\n{question}\n\nReference Answer:\n{reference or ''}"
                f"\n\nAnswer:\n{answer}")
    else:
        system = JUDGE_SYSTEM
        user = f"Passages:\n{passages}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=1024,
                temperature=0,
                timeout=timeout,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content or ""
            verdict = parse_verdict(content, answer)
            if verdict is not None:
                return {"ok": True, **verdict}
            last_err = "unparseable judge output"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(2 ** attempt)
    return {"ok": False, "error": last_err}


def load_cache(cache, rows):
    """加载行级缓存；question/answer 与当前数据不一致的条目视为脏缓存丢弃"""
    if not Path(cache).exists():
        return {}
    done, keep, stale = {}, [], 0
    for l in open(cache, encoding="utf-8"):
        r = json.loads(l)
        i = r["idx"]
        cur = rows[i] if 0 <= i < len(rows) else None
        if (cur is not None and r.get("question") == cur["question"]
                and r.get("answer") == (cur.get("answer") or "").strip()):
            done[i] = r
            keep.append(r)
        else:
            stale += 1
    if stale:
        print(f"  WARNING: {stale} stale cache entries discarded")
        Path(cache).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                               encoding="utf-8")
    return done


def score_file(client, path, max_retries, timeout, workers, mode="faithful"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    suffix = ".nocontext" if mode == "nocontext" else ""
    cache = Path(path).stem + "_spans" + suffix + ".jsonl"
    done = load_cache(cache, rows)
    pending = [i for i in range(len(rows)) if i not in done]
    print(f"{path}: {len(rows)} rows, {len(pending)} to score (cache: {cache})")

    def work(i):
        r = rows[i]
        question = r["question"]
        answer = (r.get("answer") or "").strip()
        passages = "\n".join(r.get("contexts") or [])
        rec = {"idx": i, "question": question, "answer": answer,
               "refusal": is_refusal(answer), "hallucinated": False,
               "spans": [], "ok": True}
        if not answer or rec["refusal"]:
            return rec  # 空答/拒答：无幻觉，不调 API
        res = judge_one(client, question, passages, answer, max_retries, timeout,
                        mode=mode, reference=r.get("ground_truth"))
        rec["ok"] = res["ok"]
        rec["hallucinated"] = res.get("hallucinated", False)
        rec["spans"] = res.get("spans", [])
        if not res["ok"]:
            rec["error"] = res["error"]
        return rec

    results = dict(done)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, i): i for i in pending}
        n_ok = 0
        t0 = time.time()
        with open(cache, "a", encoding="utf-8") as f:
            for fut in as_completed(futs):
                i = futs[fut]
                rec = fut.result()
                results[i] = rec
                if rec["ok"]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                if n_ok % 10 == 0 and n_ok > 0:
                    print(f"  scored {n_ok} ({time.time()-t0:.0f}s)")
    failed = [r for r in results.values() if not r.get("ok")]
    if failed:
        print(f"  WARNING: {len(failed)} rows failed judge, skipped")
    return [results[i] for i in range(len(rows))], failed


def metrics(results):
    n = len(results)
    vals = [r for r in results if r.get("ok")]
    m = len(vals)
    non_refusal = [r for r in vals if not r["refusal"]]
    return {
        "n": n, "n_ok": m,
        "hallucination_rate": sum(r["hallucinated"] for r in vals) / max(m, 1),
        "hall_rate_non_refusal": (sum(r["hallucinated"] for r in non_refusal)
                                  / max(len(non_refusal), 1)),
        "refusal_rate": sum(r["refusal"] for r in vals) / max(m, 1),
        "n_hall": sum(r["hallucinated"] for r in vals),
        "n_refusal": sum(r["refusal"] for r in vals),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True, help="answers_*.jsonl，按参数顺序排表")
    p.add_argument("--out", default="eval_report_official.md")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--mode", choices=["faithful", "nocontext"], default="faithful",
                   help="faithful=以 passages 判（RAG 场景）；nocontext=以参考回答判事实（检索失败场景）")
    args = p.parse_args()

    client = OpenAI(api_key=load_key(), base_url=MINIMAX_OPENAI_BASE, timeout=args.timeout)
    per = {}
    failed_all = []
    for f in args.files:
        results, failed = score_file(client, f, args.max_retries, args.timeout, args.workers,
                                     mode=args.mode)
        per[f] = (results, metrics(results))
        failed_all += [(f, r["idx"], r.get("error")) for r in failed]

    lines = ["# RAGTruth 官方协议评测：回答级幻觉率对比\n"]
    lines.append(f"- judge: {MODEL}（think disabled，temperature 0，LLM-as-judge）")
    lines.append(f"- 口径: 回答级二元判定（含/不含幻觉），span 仅作示例证据，不参与指标——"
                 f"对齐 RAGTruth 官方 response-level 口径\n")
    heads = [f.split("/")[-1].replace("answers_", "").replace(".jsonl", "") for f in args.files]
    lines += [f"| 指标 | " + " | ".join(heads) + " |",
              f"|---|" + "---|" * len(heads)]
    lines.append(f"| 幻觉率（含幻觉回答占比） | "
                 + " | ".join(f"{per[f][1]['hallucination_rate']:.3f}" for f in args.files) + " |")
    lines.append(f"| 幻觉率（仅非拒答回答） | "
                 + " | ".join(f"{per[f][1]['hall_rate_non_refusal']:.3f}" for f in args.files) + " |")
    lines.append(f"| 拒答率 | "
                 + " | ".join(f"{per[f][1]['refusal_rate']:.3f}" for f in args.files) + " |")
    lines += ["", f"N = {per[args.files[0]][1]['n_ok']}（每模型，失败行不计入）", ""]

    lines += ["## 每问明细\n", "| # | question | " + " | ".join(heads) + " |"]
    lines.append("|---|----------|" + "---|" * len(heads))
    first = per[args.files[0]][0]
    for i, r in enumerate(first):
        q = r["question"][:60].replace("|", "\\|")
        cells = []
        for f in args.files:
            rr = per[f][0][i]
            cells.append("有幻觉" if rr.get("hallucinated") else "-")
        lines.append(f"| {i+1} | {q} | " + " | ".join(cells) + " |")

    lines += ["", "## 幻觉示例（每模型前 2 条被判为含幻觉的回答）"]
    for f in args.files:
        lines += ["", f"### {f}", ""]
        shown = 0
        for r in per[f][0]:
            if r.get("hallucinated") and r.get("ok"):
                lines.append(f"- Q: {r['question'][:80]}")
                lines.append(f"  A: {r['answer'][:150]}...")
                for s in r["spans"][:3]:
                    lines.append(f"  - [{s['label_type']}] \"{s['text'][:80]}\"")
                shown += 1
                if shown >= 2:
                    break
        if shown == 0:
            lines.append("- 无")

    if failed_all:
        lines += ["", "## 未完成行（下次运行自动重试）"]
        for f, i, err in failed_all:
            lines.append(f"- {f} #{i}: {err}")

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved -> {args.out}")
    for f in args.files:
        m = per[f][1]
        print(f"{f}: hall_rate={m['hallucination_rate']:.3f} "
              f"({m['n_hall']}/{m['n_ok']}) refusal={m['refusal_rate']:.3f} "
              f"({m['n_refusal']})")


if __name__ == "__main__":
    main()
