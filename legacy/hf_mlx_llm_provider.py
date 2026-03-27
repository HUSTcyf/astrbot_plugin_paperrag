"""
HuggingFace MLX LLM Provider for PaperRAG
直接从 Hugging Face 加载 qwen3-8b-4bit 模型（纯文本）

使用方式:
    1. 安装依赖: pip install mlx-lm
    2. 在 PaperRAG 配置中设置 hf_llm_model_path 为 "mlx-community/Qwen3-8B-4bit"
"""

import time
from typing import Any, List, Optional

from astrbot.api import logger


class HFMlxLLMProvider:
    """HuggingFace MLX LLM Provider

    基于 Apple MLX 框架的 LLM Provider，专门用于 Apple Silicon Mac
    支持直接从 HuggingFace 加载 mlx 优化模型（纯文本）
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3-8B-4bit",
        max_tokens: int = 512,
        temperature: float = 0.7,
    ):
        """
        Args:
            model_path: HuggingFace 模型路径或本地路径
                       推荐: "mlx-community/Qwen3-8B-4bit"
            max_tokens: 最大生成token数
            temperature: 生成温度
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.model_name = model_path
        self._initialized = False

    async def initialize(self) -> None:
        """初始化模型和分词器（异步）"""
        if self._initialized:
            logger.info("[HF-MLX-LLM] 模型已初始化，跳过")
            return

        logger.info(f"[HF-MLX-LLM] 正在加载模型: {self.model_path}")

        start_time = time.time()

        try:
            from mlx_lm import load

            # 加载模型和分词器
            logger.info("[HF-MLX-LLM] 加载 MLX 模型和分词器...")
            self.model, self.tokenizer = load(
                self.model_path,
                tokenizer_config={
                    "trust_remote_code": True,
                }
            )
            logger.info("[HF-MLX-LLM] 模型和分词器加载完成")

            # 评估模式
            self.model.eval()

            elapsed = time.time() - start_time
            logger.info(f"[HF-MLX-LLM] ✅ 模型初始化完成，耗时: {elapsed:.2f}秒")
            self._initialized = True

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HF-MLX-LLM] ❌ 模型加载失败: {e}，耗时: {elapsed:.2f}秒")
            import traceback
            logger.error(f"[HF-MLX-LLM] 详细错误: {traceback.format_exc()}")
            raise

    async def chat(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> "LLMResponse":
        """
        纯文本聊天接口

        Args:
            prompt: 提示词
            temperature: 生成温度

        Returns:
            LLMResponse 对象
        """
        if not self._initialized:
            await self.initialize()

        temp = temperature if temperature is not None else self.temperature

        logger.info(f"[HF-MLX-LLM] 开始推理，prompt长度: {len(prompt)}")

        start_time = time.time()

        try:
            from mlx_lm.generate import generate

            # 使用 mlx_lm.generate 生成
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=temp,
            )

            elapsed = time.time() - start_time
            logger.info(f"[HF-MLX-LLM] ✅ 推理完成，耗时: {elapsed:.2f}秒")
            logger.debug(f"[HF-MLX-LLM] 生成的响应: {response[:200]}...")

            return LLMResponse(
                content=response,
                model=self.model_name,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[HF-MLX-LLM] ❌ 推理失败，耗时: {elapsed:.2f}秒，错误: {e}")
            import traceback
            logger.error(f"[HF-MLX-LLM] 详细错误: {traceback.format_exc()}")
            raise

    async def test(self) -> bool:
        """测试连接"""
        try:
            logger.info("[HF-MLX-LLM] 执行测试...")
            await self.initialize()
            result = await self.chat(prompt="Hello, respond with only 'PONG'")
            success = "PONG" in result.content
            logger.info(f"[HF-MLX-LLM] 测试{'成功' if success else '失败'}: {result.content}")
            return success
        except Exception as e:
            logger.error(f"[HF-MLX-LLM] 测试失败: {e}")
            return False


class LLMResponse:
    """简化的 LLM 响应类"""

    def __init__(self, content: str, model: str = ""):
        self.content = content
        self.model = model
        self.extra_kwargs = {}

    def __repr__(self):
        return f"LLMResponse(content='{self.content[:100]}...', model='{self.model}')"


# 单例实例
_llm_provider_instance: Optional[HFMlxLLMProvider] = None


def get_hf_mlx_llm_provider(
    model_path: str = "mlx-community/Qwen3-8B-4bit",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> HFMlxLLMProvider:
    """
    获取 HF MLX LLM Provider 单例

    Args:
        model_path: HuggingFace 模型路径
        max_tokens: 最大token数
        temperature: 温度

    Returns:
        HFMlxLLMProvider 实例
    """
    global _llm_provider_instance

    if _llm_provider_instance is None:
        logger.info(f"[HF-MLX-LLM] 创建新 Provider 实例: {model_path}")
        _llm_provider_instance = HFMlxLLMProvider(
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    return _llm_provider_instance


def reset_hf_mlx_llm_provider() -> None:
    """重置单例（用于重新加载模型）"""
    global _llm_provider_instance
    _llm_provider_instance = None
    logger.info("[HF-MLX-LLM] Provider 单例已重置")
