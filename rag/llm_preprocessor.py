from typing import Dict


def remove_reference_sections(text: str, ref_sections: Dict[str, str]) -> str:
    """从原文中移除参考文献部分和页码标记"""
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        # 删除页码标记 [Page X]
        if stripped.startswith('[Page ') and stripped.endswith(']'):
            continue
        if stripped.lower() == 'references':
            break  # 找到 References 后删除其及之后所有行
        result.append(line)
    return '\n'.join(result)
