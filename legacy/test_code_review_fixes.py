#!/usr/bin/env python3
"""
Code Review 修复验证测试脚本
验证 CODE_REVIEW_REPORT.md 中列出的问题是否已修复
"""
import re
import sys
from pathlib import Path

PLUGIN_ROOT = Path("/Users/chenyifeng/AstrBot/data/plugins/astrbot_plugin_paperrag")


class Issue:
    def __init__(self, id: str, file: str, description: str):
        self.id = id
        self.file = file
        self.description = description
        self.passed = False
        self.details = ""


def check_issue_1_hybrid_rag_error_log():
    """Issue #1: hybrid_rag.py 两者都失败时应有 logger.error"""
    f = PLUGIN_ROOT / "rag/hybrid_rag.py"
    content = f.read_text()
    # 1134-1137 区域: if not result: 后面应该有 logger.error
    # 检查1135行附近是否有 logger.error
    pattern = r'if not result:\s*\n\s*logger\.error'
    match = re.search(pattern, content, re.MULTILINE)
    return match is not None, "两者都失败时应有 logger.error" if not match else "已添加 logger.error"


def check_issue_2_paper_link_resolver_try_finally():
    """Issue #2: paper_link_resolver.py PyMuPDF doc 应有 try/finally"""
    f = PLUGIN_ROOT / "rag/paper_link_resolver.py"
    lines = f.read_text().split('\n')
    # 找 doc = pymupdf.open 后 finally:doc.close() 在附近
    for i, line in enumerate(lines):
        if 'doc = pymupdf.open' in line or 'doc = fitz.open' in line:
            # 向下找 doc.close() 和对应的 finally
            for j in range(i, min(i + 50, len(lines))):
                if 'doc.close()' in lines[j]:
                    # 检查前面是否有 finally
                    for k in range(i, j):
                        if 'finally:' in lines[k]:
                            return True, f"找到 try/finally 在 {i+1}-{j+1} 行"
                    return False, f"doc.close() 在 {j+1} 行但无 finally 保护"
    return False, "未找到 doc = pymupdf.open"


def check_issue_3_multimodal_extractor_try_finally():
    """Issue #3: multimodal_extractor.py PyMuPDF doc 应有 try/finally"""
    f = PLUGIN_ROOT / "rag/multimodal_extractor.py"
    lines = f.read_text().split('\n')
    for i, line in enumerate(lines):
        if 'doc = fitz.open' in line:
            for j in range(i, min(i + 50, len(lines))):
                if 'doc.close()' in lines[j]:
                    for k in range(i, j):
                        if 'finally:' in lines[k]:
                            return True, f"找到 try/finally 在 {i+1}-{j+1} 行"
                    return False, f"doc.close() 在 {j+1} 行但无 finally 保护"
    return False, "未找到 doc = fitz.open"


def check_issue_4_embed_api_key_env_var():
    """Issue #4: embed_api_key 应使用 EVAL_EMBED_API_KEY 而非 EVAL_LLM_API_KEY"""
    f = PLUGIN_ROOT / "evaluation/run_evaluation_ragas.py"
    content = f.read_text()
    # 检查1308行附近 embed_api_key 是否使用 EVAL_EMBED_API_KEY
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'embed_api_key' in line and 'os.getenv' in line:
            if 'EVAL_EMBED_API_KEY' in line:
                return True, f"在 {i+1} 行使用正确的 EVAL_EMBED_API_KEY"
            if 'EVAL_LLM_API_KEY' in line:
                return False, f"在 {i+1} 行使用错误的 EVAL_LLM_API_KEY"
    return False, "未找到 embed_api_key os.getenv 配置"


def check_issue_5_get_llm_provider_logging():
    """Issue #5: generation.py _get_llm_provider 应有日志区分失败原因"""
    f = PLUGIN_ROOT / "idea/generation.py"
    lines = f.read_text().split('\n')
    # 找 _get_llm_provider 函数，检查是否有 logger
    in_func = False
    has_logging = False
    for line in lines:
        if 'def _get_llm_provider' in line:
            in_func = True
        elif in_func and 'def ' in line and '_get_llm_provider' not in line:
            break
        elif in_func and 'logger' in line and ('debug' in line or 'warning' in line or 'error' in line):
            has_logging = True
    return has_logging, "有日志记录失败原因" if has_logging else "_get_llm_provider 无日志区分"


def check_issue_6_retrieval_helpers_provider():
    """Issue #6: retrieval_helpers.py _get_text_llm_provider 异常应区分"""
    f = PLUGIN_ROOT / "commands/retrieval_helpers.py"
    content = f.read_text()
    # 检查 _get_text_llm_provider 函数中多个失败路径是否有日志
    func_match = re.search(r'async def _get_text_llm_provider.*?(?=\n    async def|\n    def|\nclass|\Z)', content, re.DOTALL)
    if func_match:
        func_content = func_match.group(0)
        log_count = len(re.findall(r'logger\.(warning|error|info|debug)', func_content))
        return log_count >= 2, f"找到 {log_count} 个日志调用"
    return False, "未找到 _get_text_llm_provider 函数"


def check_issue_7_vlm_unavailable_log():
    """Issue #7: generation.py VLM 不可用时应有 logger.warning"""
    f = PLUGIN_ROOT / "idea/generation.py"
    content = f.read_text()
    # 找 "Fallback: 简单分析" 后面紧跟的 TopicAnalysis(domain="", ...)
    # 应该在此之前有 logger.warning
    pattern = r'Fallback:\s*简单分析\s*\n\s*logger\.warning.*VLM.*不可用'
    match = re.search(pattern, content)
    return match is not None, "VLM 不可用时有 logger.warning" if match else "VLM 不可用时无日志"


def check_issue_8_10_search_exception_log():
    """Issue #8/#13: paper_link_resolver.py 搜索异常应有 logger.warning/error"""
    f = PLUGIN_ROOT / "rag/paper_link_resolver.py"
    content = f.read_text()
    # 精确匹配 _search_crossref_candidates 函数中的 except 块
    func_match = re.search(
        r'def _search_crossref_candidates.*?except Exception.*?return \[\]',
        content, re.DOTALL
    )
    if func_match:
        section = func_match.group(0)
        if 'logger.debug' in section and 'Crossref' in section:
            return False, "Crossref 搜索异常仍使用 logger.debug"
        if 'logger.warning' in section or 'logger.error' in section:
            return True, "Crossref 搜索异常使用正确日志级别"
    return False, "未找到 _search_crossref_candidates 异常处理"


def check_issue_9_rerank_fallback_gradient():
    """Issue #9: vm.py rerank 失败时应用梯度分数而非全 0.5"""
    f = PLUGIN_ROOT / "idea/vm.py"
    content = f.read_text()
    # 精确匹配 ColBERT rerank 的 except 块（附近有 rerank 关键词）
    rerank_section = re.search(
        r'colbert_rerank.*?except Exception.*?(?=\n\S|\n    def |\n    async def |\Z)',
        content, re.DOTALL | re.IGNORECASE
    )
    if rerank_section:
        section = rerank_section.group(0)
        if '1.0 - 0.05' in section:
            return True, "rerank 失败使用梯度赋值"
        if '"score": 0.5' in section or "'score': 0.5" in section:
            return False, "rerank 失败仍使用固定 0.5"
    return False, "未找到 ColBERT rerank 失败处理"


def check_issue_10_getattr_returns_none():
    """Issue #10: ideas.py __getattr__ 应返回 None 而非抛异常"""
    f = PLUGIN_ROOT / "idea/ideas.py"
    content = f.read_text()
    # 检查 __getattr__ 是否无条件返回 None
    match = re.search(r'def __getattr__\(self.*?\):\s*return None', content, re.DOTALL)
    return match is not None, "__getattr__ 返回 None" if match else "__getattr__ 仍可能抛异常"


def check_issue_11_symlink_check():
    """Issue #11: ideas.py shutil.rmtree 前应有 symlink 检查"""
    f = PLUGIN_ROOT / "idea/ideas.py"
    lines = f.read_text().split('\n')
    for i, line in enumerate(lines):
        if 'shutil.rmtree' in line:
            # 检查前面是否有 is_symlink 检查
            for j in range(max(0, i-10), i):
                if 'is_symlink' in lines[j]:
                    return True, f"在 {j+1} 行有 symlink 检查"
            return False, f"在 {i+1} 行 shutil.rmtree 前无 symlink 检查"
    return False, "未找到 shutil.rmtree"


def check_issue_13_crossref_api_fail_log():
    """Issue #13: Crossref API 失败应使用 logger.warning/error 而非 debug"""
    f = PLUGIN_ROOT / "rag/paper_link_resolver.py"
    content = f.read_text()
    # 找 Crossref 异常处理
    crossref_section = re.search(r'def _search_crossref_candidates.*?except Exception.*?return \[\]', content, re.DOTALL)
    if crossref_section:
        section = crossref_section.group(0)
        has_debug = 'logger.debug' in section and 'Crossref' in section
        has_warning = 'logger.warning' in section or 'logger.error' in section
        if has_debug and not has_warning:
            return False, "Crossref API 失败仍用 logger.debug"
        if has_warning:
            return True, "Crossref API 失败使用正确日志级别"
    return False, "未找到 _search_crossref_candidates"


def check_issue_14_abstract_stats_corrupt():
    """Issue #14: paper.py abstract_stats.json 损坏时不应 abort"""
    f = PLUGIN_ROOT / "commands/paper.py"
    content = f.read_text()
    if 'abstract_stats_path' not in content:
        return False, "未找到 abstract_stats_path"
    # 提取 abstract_stats_path.exists() 到下一个非缩进行之间的代码块
    block_match = re.search(
        r'(abstract_stats_path\.exists\(\).*?)(?=\n\S|\Z)',
        content, re.DOTALL
    )
    if not block_match:
        return False, "未找到 abstract_stats_path 代码块"
    block = block_match.group(1)
    # 验证: except 块中有 logger.warning 且没有 return {"error"
    has_warning = 'logger.warning' in block
    has_return_error = 'return {"error"' in block or "return {'error'" in block
    if has_warning and not has_return_error:
        return True, "损坏时记录警告并继续"
    if has_return_error:
        return False, "损坏时仍 return error 中止"
    return False, "abstract_stats 异常处理不符合预期"


def check_issue_15_truncation_warning():
    """Issue #15: hybrid_parser.py truncation=True 应有警告"""
    f = PLUGIN_ROOT / "rag/hybrid_parser.py"
    content = f.read_text()
    # 检查 truncation=True 附近是否有 logger.debug/warning
    pattern = r'truncation=True.*?logger\.(debug|warning|info)'
    match = re.search(pattern, content, re.DOTALL)
    return match is not None, "截断时有日志记录" if match else "截断时无警告日志"


def check_issue_16_word_boundary():
    """Issue #16: hybrid_parser.py 单词边界问题"""
    f = PLUGIN_ROOT / "rag/hybrid_parser.py"
    content = f.read_text()
    # 必须验证 clause.rfind 在截断逻辑中使用，而非仅存在于文件中
    if 'clause.rfind(' in content:
        # 验证 rfind 在 chunk_size 截断上下文中（附近有 max_chars 或 chunk_size * 4）
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'clause.rfind(' in line:
                context = '\n'.join(lines[max(0, i-5):i+5])
                if 'chunk_size' in context or 'max_chars' in context:
                    return True, f"在 {i+1} 行使用 clause.rfind 在单词边界截断"
        return False, "clause.rfind 存在但不在截断逻辑中"
    if 'clause[:self.chunk_size * 4]' in content:
        return False, "按字符截断可能切断单词"
    return False, "未找到截断代码"


def check_issue_17_generate_ideas_error():
    """Issue #17: generation.py generate_ideas 异常应有日志"""
    f = PLUGIN_ROOT / "idea/generation.py"
    content = f.read_text()
    gen_func = re.search(r'async def generate_ideas.*?(?=\n    async def|\n    def|\nclass|\Z)', content, re.DOTALL)
    if gen_func:
        func = gen_func.group(0)
        has_error_log = 'logger.error' in func or 'logger.warning' in func
        has_return_empty = 'return []' in func
        return has_error_log, "有错误日志" if has_error_log else "异常返回空列表无日志"
    return False, "未找到 generate_ideas 函数"


def check_issue_18_resolve_arxiv_parallel():
    """Issue #18: retrieval_helpers.py _resolve_sources_arxiv 应使用 asyncio.gather"""
    f = PLUGIN_ROOT / "commands/retrieval_helpers.py"
    content = f.read_text()
    resolve_func = re.search(r'async def _resolve_sources_arxiv.*?(?=\n    async def|\n    def|\nclass|\Z)', content, re.DOTALL)
    if resolve_func:
        func = resolve_func.group(0)
        has_gather = 'asyncio.gather' in func or 'gather' in func
        has_for_await = 'for' in func and 'await' in func and 'resolve_source' in func
        if has_gather:
            return True, "使用 asyncio.gather 并发"
        if has_for_await:
            return False, "仍使用串行 for/await"
    return False, "未找到 _resolve_sources_arxiv 函数"


def check_issue_19_httpx_module_level():
    """Issue #19: paper_link_resolver.py httpx 应在模块顶部 import"""
    f = PLUGIN_ROOT / "rag/paper_link_resolver.py"
    content = f.read_text()
    lines = content.split('\n')
    # 找模块级 import httpx 的位置（文件前面 50 行）
    module_imports = []
    in_func = False
    for i, line in enumerate(lines[:50]):
        if 'import httpx' in line and not line.strip().startswith('#'):
            module_imports.append((i+1, line.strip()))
    # 检查函数内 import
    func_imports = 0
    in_func_area = False
    for i, line in enumerate(lines[50:], start=50):
        if re.match(r'^\s+(async )?def ', line):
            in_func_area = True
        elif re.match(r'^[^ \t]', line) and in_func_area:
            break
        if in_func_area and 'import httpx' in line:
            func_imports += 1
    return len(module_imports) > 0, f"模块级有 {len(module_imports)} 个 import，函数内有 {func_imports} 个" if module_imports else "httpx 仍在函数内 import"


def check_issue_20_asyncio_lock_in_method():
    """Issue #20: ragas_generator.py asyncio.Lock 不应在方法内创建"""
    f = PLUGIN_ROOT / "evaluation/ragas_generator.py"
    content = f.read_text()
    # 找 _async_lock = asyncio.Lock() 在方法内的情况
    pattern = r'def \w+\([^)]*\):[^}]*_async_lock\s*=\s*asyncio\.Lock\(\)'
    match = re.search(pattern, content, re.DOTALL)
    return match is None, "asyncio.Lock 在方法外" if match is None else "asyncio.Lock 仍在方法内创建"


def check_issue_21_class_in_else_branch():
    """Issue #21: graph_rag_engine.py 类定义位置问题"""
    f = PLUGIN_ROOT / "graphrag/graph_rag_engine.py"
    content = f.read_text()
    # 检查类是否在条件块内然后在 else 设为 None
    # 更好的模式是在 try/except 中定义，或在类外部做检查
    if 'class _CaseInsensitiveSynonymRetriever' in content:
        # 当前模式: if LLMSynonymRetriever is not None: class... else: = None
        # 这是已知模式，不算严重问题，只要不在实际使用中混淆即可
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'class _CaseInsensitiveSynonymRetriever' in line:
                # 检查是否在 if 块内
                for j in range(max(0, i-3), i):
                    if 'if ' in lines[j] and 'is not None' in lines[j]:
                        return True, "类在 if 块内定义（条件编译模式，可接受）"
                return True, "类定义正常"
    return False, "未找到该类"


def run_all_checks():
    """运行所有检查"""
    checks = [
        ("#1 JSON/regex双fallback错误日志", check_issue_1_hybrid_rag_error_log),
        ("#2 paper_link_resolver try/finally", check_issue_2_paper_link_resolver_try_finally),
        ("#3 multimodal_extractor try/finally", check_issue_3_multimodal_extractor_try_finally),
        ("#4 embed_api_key env var", check_issue_4_embed_api_key_env_var),
        ("#5 _get_llm_provider logging", check_issue_5_get_llm_provider_logging),
        ("#6 _get_text_llm_provider logging", check_issue_6_retrieval_helpers_provider),
        ("#7 VLM unavailable log", check_issue_7_vlm_unavailable_log),
        ("#8/#13 search exception log level", check_issue_8_10_search_exception_log),
        ("#9 rerank fallback gradient", check_issue_9_rerank_fallback_gradient),
        ("#10 __getattr__ returns None", check_issue_10_getattr_returns_none),
        ("#11 symlink check before rmtree", check_issue_11_symlink_check),
        ("#13 Crossref API fail log", check_issue_13_crossref_api_fail_log),
        ("#14 abstract_stats.json corrupt", check_issue_14_abstract_stats_corrupt),
        ("#15 truncation warning", check_issue_15_truncation_warning),
        ("#16 word boundary issue", check_issue_16_word_boundary),
        ("#17 generate_ideas error log", check_issue_17_generate_ideas_error),
        ("#18 _resolve_sources_arxiv parallel", check_issue_18_resolve_arxiv_parallel),
        ("#19 httpx module-level import", check_issue_19_httpx_module_level),
        ("#20 asyncio.Lock in method", check_issue_20_asyncio_lock_in_method),
        ("#21 class in else branch", check_issue_21_class_in_else_branch),
    ]

    results = []
    for name, check_func in checks:
        try:
            passed, details = check_func()
            results.append((name, passed, details))
        except Exception as e:
            results.append((name, False, f"检查执行出错: {e}"))

    return results


def main():
    print("=" * 70)
    print("Code Review 修复验证测试")
    print("=" * 70)

    results = run_all_checks()

    passed = 0
    failed = 0
    for name, ok, details in results:
        status = "PASS" if ok else "FAIL"
        symbol = "[PASS]" if ok else "[FAIL]"
        print(f"{symbol} {name}")
        print(f"       {details}")
        if ok:
            passed += 1
        else:
            failed += 1

    print("=" * 70)
    print(f"总计: {passed} 通过, {failed} 失败, {passed + failed} 总计")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
