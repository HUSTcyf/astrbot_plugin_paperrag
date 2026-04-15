"""
Ideas CRUD 管理与 Topic/Context 管理
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .datatypes import ResearchIdea

from astrbot.api import logger

from .utils import topic_hash, fuse_knowledge_context, IdeaEngineUtils

class IdeaEngineIdeas(IdeaEngineUtils):
    """
    Ideas CRUD 与 Topic/Context 管理。

    继承链：IdeaEngineBase → IdeaEngineUtils → IdeaEngineIdeas → ... → IdeaEngine
    下游类（IdeaEngineGeneration 等）的方法通过 MRO 访问。
    IDE 静态分析无法跨 MRO 解析，运行时行为不受影响。
    """

    def __getattr__(self, name: str):
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _load_figure_captions(self, image_path: str) -> Dict[str, str]:
        """
        从 captions JSON 加载指定图片的 caption。

        Args:
            image_path: 图片完整路径，如 /.../data/figures/2502.12138v4(nopo)/14-Figure1.png

        Returns:
            Dict: key = 图片文件名 (如 "14-Figure1.png"), value = caption 文本
                空 dict 表示文件不存在或解析失败
        """
        import json
        path = Path(image_path)
        if not path.exists():
            return {}
        # 从路径提取 paper_name: .../figures/{paper_name}/{N-FigureM}.png
        figures_dir = path.parent
        paper_name = figures_dir.name  # 如 "2502.12138v4(nopo)"
        caption_file = figures_dir.parent.parent / "captions" / f"{paper_name}.json"
        if not caption_file.exists():
            return {}
        try:
            with open(caption_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # data: { "14-Figure5": {"caption": "...", "filename": "14-Figure1.png", ...}, ... }
            # 建立 filename -> caption 的反向索引
            fname_to_caption = {}
            for v in data.values():
                fname = v.get("filename", "")
                caption = v.get("caption", "")
                if fname and caption:
                    fname_to_caption[fname] = caption
            return fname_to_caption
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[IdeaEngine] 读取 caption 文件失败: {caption_file} ({e})")
            return {}

    def _topic_folder(self, topic: str) -> Path:
        """获取 topic 对应的文件夹路径（使用 MD5 哈希，跨进程稳定）"""
        return self._get_ideas_dir() / self._topic_hash(topic)

    def _topic_hash(self, topic: str) -> str:
        """计算 topic 对应的 folder hash（MD5 hex 前16位）"""
        return hashlib.md5(topic.encode()).hexdigest()[:16]

    def _get_topic_index(self) -> Dict[str, str]:
        """获取 folder_name → topic 的索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    logger.warning("[IdeaEngine] topic_index.json 格式错误（非 dict）")
            except (json.JSONDecodeError, IOError):
                pass
        return {}

    def _save_topic_index(self, index: Dict[str, str]) -> None:
        """保存 topic → folder_name 索引"""
        index_file = self._get_ideas_dir() / "topic_index.json"
        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

    def list_all_topics(self) -> List[Dict[str, Any]]:
        """列出所有已保存的 topic 及其元信息"""
        index = self._get_topic_index()
        ideas_dir = self._get_ideas_dir()
        result = []

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue

            idea_files = [f for f in folder.glob("*.json") if f.name != "context.json"]
            created_at = ""
            if (folder / "context.json").exists():
                try:
                    with open(folder / "context.json", "r", encoding="utf-8") as f:
                        ctx = json.load(f)
                        if isinstance(ctx, dict):
                            created_at = ctx.get("created_at", "")
                except (json.JSONDecodeError, IOError):
                    pass

            result.append({
                "topic": topic,
                "folder": folder_name,
                "idea_count": len(idea_files),
                "created_at": created_at
            })

        return result

    def _topic_folder_by_hash(self, folder_hash: str) -> Path:
        """根据 folder hash 获取文件夹路径"""
        return self._get_ideas_dir() / folder_hash

    def _get_context_path(self, topic: str) -> Path:
        """获取 topic 文件夹下的 context.json 路径"""
        return self._topic_folder(topic) / "context.json"

    def _save_context(self, topic: str, knowledge: Dict[str, Any]) -> None:
        """保存共享 context 到 topic 文件夹"""
        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])
        logger.info(f"[IdeaEngine] _save_context 开始:")
        logger.info(f"  - folder: {folder}")
        logger.info(f"  - local_results 数量: {len(local_results)}")
        logger.info(f"  - web_results 数量: {len(web_results)}")
        for i, lr in enumerate(local_results):
            text = lr.get("text", "")
            paper = lr.get("paper", "?")[:30]
            metadata = lr.get("metadata", {})
            img_path = metadata.get("image_path", "")
            table_path = metadata.get("table_csv_path", "")
            logger.info(f"    local_result[{i}]: paper={paper}, text_len={len(text)}, img={bool(img_path)}, table={bool(table_path)}")

        ctx_data = {
            "topic": topic,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "local_results": local_results,
            "web_results": web_results
        }
        logger.info(f"[IdeaEngine] _save_context: 准备写入 {len(ctx_data['local_results'])} 条 local_results")
        with open(self._get_context_path(topic), "w", encoding="utf-8") as f:
            json.dump(ctx_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[IdeaEngine] _save_context: 写入完成，文件大小={self._get_context_path(topic).stat().st_size} bytes")

    def _load_context(self, topic: str) -> Optional[Dict[str, Any]]:
        """加载共享 context（topic 可能是原始名称或 folder hash）"""
        folder = self._get_ideas_dir() / topic
        if not folder.exists():
            folder = self._topic_folder(topic)
        ctx_path = folder / "context.json"
        if not ctx_path.exists():
            return None
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    logger.info(f"[IdeaEngine] _load_context: local_results={len(data.get('local_results', []))}")
                    return data
                logger.warning(f"[IdeaEngine] context.json 格式错误（非 dict 类型）: {type(data)}")
                return None
        except (json.JSONDecodeError, IOError):
            return None

    # ==================== Ideas CRUD ====================

    def save_ideas_to_file(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any]
    ) -> List[Tuple[str, Path]]:
        """
        将多个想法及上下文保存到 topic 文件夹

        目录结构:
        ideas/
          topic_index.json
          <hash(topic)>/
            context.json          # 共享 context
            <uuid1>.json        # 单个 idea
            <uuid2>.json
        """
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        local_results = knowledge.get("local_results", [])
        web_results = knowledge.get("web_results", [])
        logger.info(f"[IdeaEngine] save_ideas_to_file 开始保存:")
        logger.info(f"  - topic: {topic}")
        logger.info(f"  - ideas 数量: {len(ideas)}")
        logger.info(f"  - local_results 数量: {len(local_results)}")
        logger.info(f"  - web_results 数量: {len(web_results)}")

        self._save_context(topic, knowledge)

        results = []
        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)
            results.append((idea_uuid, file_path))
            logger.info(f"[IdeaEngine] 想法已保存: {file_path}")

        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

        return results

    def load_ideas_by_topic(
        self, folder_hash: str
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """根据 folder hash 加载该 topic 下所有想法"""
        folder = self._get_ideas_dir() / folder_hash
        if not folder.exists():
            logger.warning(f"[IdeaEngine] 未找到 folder_hash={folder_hash} 的文件夹")
            return [], None

        loaded = []
        for file_path in folder.glob("*.json"):
            if file_path.name == "context.json":
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    loaded.append(data)
                else:
                    logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {file_path.name}")
            except (json.JSONDecodeError, IOError):
                logger.warning(f"[IdeaEngine] 跳过损坏的想法文件: {file_path.name}")

        context_data = self._load_context_by_folder(folder_hash)
        return loaded, context_data

    def _load_context_by_folder(self, folder_hash: str) -> Optional[Dict[str, Any]]:
        """根据 folder_hash 加载 context"""
        folder = self._get_ideas_dir() / folder_hash
        ctx_path = folder / "context.json"
        if not ctx_path.exists():
            return None
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                return None
        except (json.JSONDecodeError, IOError):
            return None

    def load_ideas_by_uuids(
        self,
        uuids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """根据 UUID 列表加载想法，同时加载共享 context"""
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()
        loaded = []
        found_topic = None
        found_folder = None

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue
            for uid in uuids:
                file_path = folder / f"{uid}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, dict):
                            loaded.append(data)
                            if found_topic is None:
                                found_topic = topic
                                found_folder = folder_name
                        else:
                            logger.warning(f"[IdeaEngine] 想法文件格式错误（非 dict）: {uid}")
                    except (json.JSONDecodeError, IOError):
                        logger.warning(f"[IdeaEngine] 跳过损坏的想法文件: {uid}")

        context_data = None
        if found_folder:
            context_data = self._load_context_by_folder(found_folder)
            if context_data is None:
                context_data = {}
            context_data["_folder_hash"] = found_folder

        return loaded, context_data

    def delete_ideas_by_uuids(self, uuids: List[str]) -> Tuple[List[str], Optional[str]]:
        """根据 UUID 列表删除想法文件"""
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()
        deleted = []
        found_topic = None

        for folder_name, topic in index.items():
            folder = ideas_dir / folder_name
            if not folder.exists():
                continue
            for uid in uuids:
                file_path = folder / f"{uid}.json"
                if file_path.exists():
                    file_path.unlink()
                    deleted.append(uid)
                    if found_topic is None:
                        found_topic = topic
                    logger.info(f"[IdeaEngine] 已删除想法: {file_path}")

        return deleted, found_topic

    def clear_ideas_by_topic(self, topic: str) -> Tuple[int, Optional[str]]:
        """清空指定 topic 下所有想法（保留 folder）"""
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()

        folder_name = topic
        if topic not in index:
            folder_hash = topic_hash(topic)
            if folder_hash in index:
                folder_name = folder_hash
            else:
                return 0, None

        folder = ideas_dir / folder_name
        if not folder.exists():
            return 0, None

        json_files = list(folder.glob("*.json"))
        deleted_count = 0
        for f in json_files:
            f.unlink()
            deleted_count += 1
            logger.info(f"[IdeaEngine] 已删除想法文件: {f}")

        actual_topic = index.get(folder_name, folder_name)
        logger.info(f"[IdeaEngine] 已清空 topic「{actual_topic}」，删除 {deleted_count} 个想法文件")

        return deleted_count, actual_topic

    def delete_topic_by_hash(self, topic_or_hash: str) -> Tuple[bool, str, str]:
        """完全删除指定 topic（folder + 所有内容），包括 folder 本身"""
        ideas_dir = self._get_ideas_dir()
        index = self._get_topic_index()

        folder_name = topic_or_hash
        actual_topic = topic_or_hash
        found = False

        if topic_or_hash in index:
            actual_topic = index[topic_or_hash]
            folder_name = topic_or_hash
            found = True
        else:
            for fh, tp in index.items():
                if tp == topic_or_hash:
                    folder_name = fh
                    actual_topic = tp
                    found = True
                    break

        if not found:
            return False, topic_or_hash, ""

        folder = ideas_dir / folder_name
        if not folder.exists():
            return False, actual_topic, folder_name

        shutil.rmtree(folder)
        logger.info(f"[IdeaEngine] 已删除 topic folder: {folder}")

        if folder_name in index:
            del index[folder_name]
            self._save_topic_index(index)

        return True, actual_topic, folder_name

    def _save_ideas_append(
        self,
        ideas: List["ResearchIdea"],
        topic: str,
        knowledge: Dict[str, Any]
    ) -> List[Tuple[str, Path]]:
        """追加保存想法到已有 topic 文件夹（不覆盖已有想法）"""
        import uuid as uuid_module

        folder = self._topic_folder(topic)
        folder.mkdir(parents=True, exist_ok=True)

        index = self._get_topic_index()
        index[folder.name] = topic
        self._save_topic_index(index)

        results = []
        for idea in ideas:
            idea_uuid = str(uuid_module.uuid4())[:8]
            idea_data = {
                "id": idea_uuid,
                "topic": topic,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "idea": {
                    "title": idea.title,
                    "description": idea.description,
                    "novelty": idea.novelty,
                    "methodology": idea.methodology,
                    "potential_challenges": idea.potential_challenges,
                    "related_work": idea.related_work,
                    "feasibility": idea.feasibility,
                    "inspiration_sources": idea.inspiration_sources
                }
            }
            file_path = folder / f"{idea_uuid}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(idea_data, f, ensure_ascii=False, indent=2)
            results.append((idea_uuid, file_path))
            logger.info(f"[IdeaEngine] 追加想法已保存: {file_path}")

        return results
