"""
Llama.cpp VLM Provider for PaperRAG
使用 llama-cpp-python 实现本地视觉语言模型推理

异步架构：Llama 对象常驻 + 异步调用
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Dict

from astrbot.api import logger


class LLMResponse:
    """简化的 LLM 响应类"""
    def __init__(self, content: str, model: str = ""):
        self.content = content
        self.model = model
        self.extra_kwargs: Dict[str, Any] = {}

    def __repr__(self):
        return f"LLMResponse(content='{self.content[:100]}...', model='{self.model}')"


class LlamaCppVLMProvider:
    """
    Llama.cpp VLM Provider

    使用 llama-cpp-python 实现：
    1. 模型常驻：Llama 对象创建一次，反复使用
    2. 异步调用：使用 asyncio.run_in_executor 避免阻塞事件循环
    3. 自动重载：进程崩溃后自动重新初始化
    """

    def __init__(
        self,
        model_path: str = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
        mmproj_path: str = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = 99,
        max_tokens: int = 2560,
        temperature: float = 0.7,
        n_parallel: int = 1,
    ):
        """
        Args:
            model_path: GGUF 模型文件路径
            mmproj_path: mmproj 视觉编码器文件路径
            n_ctx: 上下文窗口大小
            n_gpu_layers: 加载到 GPU 的层数（99=全部）
            max_tokens: 最大生成 token 数
            temperature: 生成温度
            n_parallel: 并行推理数（batch 模式）
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n_parallel = n_parallel

        self._llama: Optional[Any] = None
        self._initialized = False
        self._start_time: Optional[float] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """初始化 Llama 对象（支持 9B→4B 自动降级）"""
        if self._initialized:
            logger.info("[Llama.cpp-VLM] 已初始化")
            return

        logger.info("[Llama.cpp-VLM] 开始初始化...")
        logger.info(f"  原始配置: {self.model_path}")
        logger.info(f"  n_ctx: {self.n_ctx}, n_gpu_layers: {self.n_gpu_layers}")

        # 尝试初始化，9B 失败则降级到 4B
        await self._try_initialize_with_fallback()

        self._initialized = True
        self._start_time = time.time()
        logger.info(f"[Llama.cpp-VLM] ✅ 初始化完成，当前使用: {self.model_path}")

    # 9B → 4B 降级配置
    FALLBACK_MODELS = [
        {
            "model_name": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
            "mmproj_name": "mmproj-BF16.gguf",
            "repo_id": "unsloth/Qwen3.5-9B-GGUF",
            "subdir": "Qwen3.5-9B-GGUF",
        },
        {
            "model_name": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
            "mmproj_name": "mmproj-BF16.gguf",
            "repo_id": "unsloth/Qwen3.5-4B-GGUF",
            "subdir": "Qwen3.5-4B-GGUF",
        },
    ]

    def _get_plugin_models_dir(self) -> Path:
        """获取插件的 models 目录"""
        # __file__ = .../astrbot_plugin_paperrag/llama_cpp_vlm_provider.py
        # .parent = .../astrbot_plugin_paperrag/
        return Path(__file__).parent.resolve() / "models"

    async def _ensure_models_downloaded(self) -> None:
        """检查模型文件是否存在，不存在则自动下载（9B优先，4B备用）"""
        from huggingface_hub import hf_hub_download

        models_base_dir = self._get_plugin_models_dir()

        # 定义模型信息
        model_9b = self.FALLBACK_MODELS[0]  # 9B 模型配置
        model_4b = self.FALLBACK_MODELS[1]   # 4B 模型配置

        # 构建路径
        model_9b_path = models_base_dir / model_9b["subdir"] / model_9b["model_name"]
        mmproj_9b_path = models_base_dir / model_9b["subdir"] / model_9b["mmproj_name"]
        model_4b_path = models_base_dir / model_4b["subdir"] / model_4b["model_name"]
        mmproj_4b_path = models_base_dir / model_4b["subdir"] / model_4b["mmproj_name"]

        # 处理 9B 模型（检查/下载）
        if not (model_9b_path.exists() and mmproj_9b_path.exists()):
            logger.info(f"[Llama.cpp-VLM] 9B 模型不存在，正在下载...")
            await self._download_model(
                repo_id=model_9b["repo_id"],
                model_name=model_9b["model_name"],
                mmproj_name=model_9b["mmproj_name"],
                download_dir=str(models_base_dir / model_9b["subdir"]),
            )
        else:
            logger.info(f"[Llama.cpp-VLM] 9B 模型已存在")

        # 处理 4B 模型（检查/下载）
        if not (model_4b_path.exists() and mmproj_4b_path.exists()):
            logger.info(f"[Llama.cpp-VLM] 4B 模型不存在，正在下载...")
            await self._download_model(
                repo_id=model_4b["repo_id"],
                model_name=model_4b["model_name"],
                mmproj_name=model_4b["mmproj_name"],
                download_dir=str(models_base_dir / model_4b["subdir"]),
            )
        else:
            logger.info(f"[Llama.cpp-VLM] 4B 模型已存在")

        # 默认使用 9B 模型
        if (model_9b_path.exists() and mmproj_9b_path.exists()):
            self.model_path = str(model_9b_path)
            self.mmproj_path = str(mmproj_9b_path)
            logger.info(f"[Llama.cpp-VLM] 使用 9B 模型: {self.model_path}")
        elif (model_4b_path.exists() and mmproj_4b_path.exists()):
            self.model_path = str(model_4b_path)
            self.mmproj_path = str(mmproj_4b_path)
            logger.info(f"[Llama.cpp-VLM] 9B 不可用，使用 4B 模型: {self.model_path}")
        else:
            raise RuntimeError("所有模型都下载失败")

    async def _download_model(
        self,
        repo_id: str,
        model_name: str,
        mmproj_name: str,
        download_dir: str,
    ) -> None:
        """下载单个模型的 GGUF 和 mmproj 文件"""
        from huggingface_hub import hf_hub_download

        download_path = Path(download_dir)
        model_path = download_path / model_name
        mmproj_path = download_path / mmproj_name

        # 下载模型文件
        if not model_path.exists():
            logger.info(f"[Llama.cpp-VLM] 下载模型文件: {model_name}")
            download_path.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=model_name,
                    local_dir=download_dir,
                )
                logger.info(f"[Llama.cpp-VLM] 模型下载完成: {model_name}")
            except Exception as e:
                logger.error(f"[Llama.cpp-VLM] 模型下载失败: {e}")
                raise

        # 下载 mmproj 文件
        if not mmproj_path.exists():
            logger.info(f"[Llama.cpp-VLM] 下载 mmproj 文件: {mmproj_name}")
            download_path.mkdir(parents=True, exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=mmproj_name,
                    local_dir=download_dir,
                )
                logger.info(f"[Llama.cpp-VLM] mmproj 下载完成: {mmproj_name}")
            except Exception as e:
                logger.error(f"[Llama.cpp-VLM] mmproj 下载失败: {e}")
                raise

    async def _try_initialize_with_fallback(self) -> None:
        """尝试初始化，9B 失败则降级到 4B"""
        # 首先确保模型已下载
        await self._ensure_models_downloaded()

        # 尝试加载 9B 模型
        try:
            await self._try_load_model(self.model_path, self.mmproj_path)
            return
        except Exception as e:
            logger.warning(f"[Llama.cpp-VLM] 9B 模型加载失败: {e}")

        # 9B 失败，尝试 4B 模型
        model_4b = self.FALLBACK_MODELS[1]
        model_4b_dir = self._get_plugin_models_dir() / model_4b["subdir"]
        model_4b_path = model_4b_dir / model_4b["model_name"]
        mmproj_4b_path = model_4b_dir / model_4b["mmproj_name"]

        if model_4b_path.exists() and mmproj_4b_path.exists():
            logger.info(f"[Llama.cpp-VLM] 降级到 4B 模型...")
            try:
                await self._try_load_model(str(model_4b_path), str(mmproj_4b_path))
                self.model_path = str(model_4b_path)
                self.mmproj_path = str(mmproj_4b_path)
                return
            except Exception as e:
                logger.error(f"[Llama.cpp-VLM] 4B 模型加载也失败: {e}")

        raise RuntimeError("9B 和 4B 模型都无法加载")

    async def _try_load_model(self, model_path: str, mmproj_path: str) -> None:
        """在独立线程中加载 Llama 模型"""
        import concurrent.futures

        def _load() -> Any:
            from llama_cpp import Llama
            return Llama(
                model_path=model_path,
                mmproj=mmproj_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_parallel * 32,
                verbose=False,
            )

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            self._llama = await loop.run_in_executor(executor, _load)

    def _load_llama(self) -> Any:
        """在线程中加载 Llama 模型（同步版本，保留兼容性）"""
        from llama_cpp import Llama

        llama = Llama(
            model_path=self.model_path,
            mmproj=self.mmproj_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_parallel * 32,
            verbose=False,
        )
        return llama

    async def text_chat(
        self,
        prompt: Optional[str] = None,
        session_id: Optional[str] = None,
        image_urls: Optional[List[str]] = None,
        contexts: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """
        VLM 问答接口

        Args:
            prompt: 用户问题
            image_urls: 图片路径列表
            temperature: 生成温度

        Returns:
            LLMResponse 对象
        """
        if not self._initialized:
            await self.initialize()

        temp = temperature if temperature is not None else self.temperature

        logger.info(f"[Llama.cpp-VLM] 开始推理，prompt长度: {len(prompt) if prompt else 0}")
        logger.debug(f"[Llama.cpp-VLM] 图片数量: {len(image_urls) if image_urls else 0}")

        start_time = time.time()

        try:
            # 处理图片（支持多张）
            image_paths = []
            if image_urls is not None:
                for img_url in image_urls:
                    img_path = Path(img_url)
                    if img_path.exists():
                        image_paths.append(str(img_path.resolve()))
                        logger.debug(f"[Llama.cpp-VLM] 添加图片: {img_path}")
                    else:
                        logger.warning(f"[Llama.cpp-VLM] 图片不存在: {img_path}")

            # 构建消息
            if image_paths:
                content = [
                    {"type": "image_url", "image_url": {"url": path}} for path in image_paths
                ]
                content.append({"type": "text", "text": prompt or "Describe this image."})
            else:
                content = prompt or "Describe this image."

            # 添加系统消息（如果提供了 system_prompt）
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            # 在线程池中执行推理
            assert self._llama is not None, "Llama 未初始化"
            llama = self._llama
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llama.create_chat_completion(
                    messages=messages,
                    temperature=temp,
                    max_tokens=self.max_tokens,
                )
            )

            # 解析响应
            response_text = result["choices"][0]["message"]["content"]

            elapsed = time.time() - start_time
            logger.info(f"[Llama.cpp-VLM] ✅ 推理完成，耗时: {elapsed:.2f}秒")
            logger.debug(f"[Llama.cpp-VLM] 响应: {response_text[:200]}...")

            return LLMResponse(
                content=response_text,
                model=os.path.basename(self.model_path),
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[Llama.cpp-VLM] ❌ 推理失败，耗时: {elapsed:.2f}秒，错误: {e}")
            import traceback
            logger.error(f"[Llama.cpp-VLM] 详细错误: {traceback.format_exc()}")

            # 尝试重置模型
            await self._try_recover()
            raise

    async def _try_recover(self) -> None:
        """尝试恢复模型（如果加载失败）"""
        logger.info("[Llama.cpp-VLM] 尝试恢复模型...")
        try:
            self._llama = None
            self._initialized = False
            await self.initialize()
        except Exception as e:
            logger.error(f"[Llama.cpp-VLM] 恢复失败: {e}")

    async def test(self) -> bool:
        """测试 VLM 是否正常工作"""
        try:
            logger.info("[Llama.cpp-VLM] 执行测试...")
            await self.initialize()

            # 简单的文本测试（不需要图片）
            result = await self.text_chat(
                prompt="Hello, respond with only 'PONG'",
                image_urls=None  # 不使用图片
            )

            success = "PONG" in result.content
            logger.info(f"[Llama.cpp-VLM] 测试{'成功' if success else '失败'}: {result.content}")
            return success
        except Exception as e:
            logger.error(f"[Llama.cpp-VLM] 测试失败: {e}")
            return False


# ============================================================================
# ============================================================================
# 单例模式管理
# ============================================================================

# 全局 Provider 单例
_vlm_provider_instance: Optional[LlamaCppVLMProvider] = None


def init_llama_cpp_vlm_provider(
    model_path: str,
    mmproj_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = 99,
    max_tokens: int = 2560,
    temperature: float = 0.7,
    n_parallel: int = 1,
) -> LlamaCppVLMProvider:
    """
    初始化 Llama.cpp VLM Provider 单例（应用启动时调用一次）

    Args:
        model_path: GGUF 模型文件路径
        mmproj_path: mmproj 视觉编码器文件路径
        n_ctx: 上下文大小
        n_gpu_layers: GPU 层数
        max_tokens: 最大 token 数
        temperature: 温度
        n_parallel: 并行数

    Returns:
        LlamaCppVLMProvider 实例
    """
    global _vlm_provider_instance

    logger.info(f"[Llama.cpp-VLM] 初始化 Provider: model={model_path}")
    _vlm_provider_instance = LlamaCppVLMProvider(
        model_path=model_path,
        mmproj_path=mmproj_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        max_tokens=max_tokens,
        temperature=temperature,
        n_parallel=n_parallel,
    )
    return _vlm_provider_instance


def get_llama_cpp_vlm_provider() -> LlamaCppVLMProvider:
    """
    获取 Llama.cpp VLM Provider 单例

    返回已初始化的单例实例。如果尚未初始化，则创建默认配置的实例。

    Returns:
        LlamaCppVLMProvider 实例
    """
    global _vlm_provider_instance

    if _vlm_provider_instance is None:
        # 未初始化时，使用默认路径创建实例
        logger.info("[Llama.cpp-VLM] 单例未初始化，创建默认实例")
        from pathlib import Path
        plugin_dir = Path(__file__).parent
        model_path = str(plugin_dir / "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf")
        mmproj_path = str(plugin_dir / "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf")
        _vlm_provider_instance = LlamaCppVLMProvider(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=4096,
            n_gpu_layers=99,
            max_tokens=2560,
            temperature=0.7,
            n_parallel=1,
        )

    return _vlm_provider_instance


def get_cached_llama_cpp_provider() -> Optional[LlamaCppVLMProvider]:
    """
    获取已缓存的 LlamaCppVLMProvider 单例

    Returns:
        缓存的实例，或 None
    """
    return _vlm_provider_instance


def check_llama_cpp_vlm_available(
    model_path: str = "./models/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf",
    mmproj_path: str = "./models/Qwen3.5-9B-GGUF/mmproj-BF16.gguf"
) -> bool:
    """
    检查 Llama.cpp VLM 模型文件是否存在

    Args:
        model_path: GGUF 模型文件路径
        mmproj_path: mmproj 视觉编码器文件路径

    Returns:
        True if both files exist, False otherwise
    """
    from pathlib import Path
    plugin_dir = Path(__file__).parent
    model_full_path = str((plugin_dir / model_path).resolve())
    mmproj_full_path = str((plugin_dir / mmproj_path).resolve())

    model_exists = os.path.exists(model_full_path)
    mmproj_exists = os.path.exists(mmproj_full_path)

    if not model_exists:
        logger.debug(f"[Llama.cpp-VLM] 模型文件不存在: {model_full_path}")
    if not mmproj_exists:
        logger.debug(f"[Llama.cpp-VLM] mmproj文件不存在: {mmproj_full_path}")

    return model_exists and mmproj_exists


def reset_llama_cpp_vlm_provider() -> None:
    """
    重置 Provider 单例（用于重新加载模型或清理资源）
    """
    global _vlm_provider_instance

    if _vlm_provider_instance is not None:
        if hasattr(_vlm_provider_instance, '_llama'):
            _vlm_provider_instance._llama = None
        _vlm_provider_instance._initialized = False
        logger.debug("[Llama.cpp-VLM] Provider 单例已清理")

    _vlm_provider_instance = None

    # 强制垃圾回收
    import gc
    gc.collect()

    logger.info("[Llama.cpp-VLM] Provider 单例已重置")
