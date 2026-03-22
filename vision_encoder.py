"""
视觉编码器模块
使用 SigLIP (SigLIP-2) 对图像进行编码生成向量
支持优雅降级：如果transformers不可用，自动禁用视觉编码
"""

import io
from typing import List, Optional, Union, Any
from pathlib import Path
from PIL import Image
import warnings
import logging

logger = logging.getLogger(__name__)

# 抑制 transformers 警告
warnings.filterwarnings("ignore", category=UserWarning)

# 尝试导入可选依赖
try:
    from transformers import AutoProcessor, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    # 定义类型占位符，避免类型检查错误
    torch = None  # type: ignore
    AutoProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    logger.warning(f"⚠️ transformers或torch未安装: {e}")


class VisionEncoder:
    """SigLIP 视觉编码器（支持优雅降级）"""

    # 支持的模型配置
    MODELS = {
        "siglip-base": "google/siglip-base-patch16-224",  # 400M参数，平衡性能和速度
        "siglip-large": "google/siglip-large-patch16-384",  # 更高精度
        "siglip-so400m": "google/siglip-so400m-patch14-384",  # 推荐
    }

    def __init__(self,
                 model_name: str = "siglip-so400m",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 fallback_to_text: bool = True):
        """
        初始化 SigLIP 视觉编码器

        Args:
            model_name: 模型名称 (siglip-base/siglip-large/siglip-so400m)
            device: 设备 (cuda/cpu/auto)，None表示自动检测
            cache_dir: 模型缓存目录
            fallback_to_text: 如果transformers不可用，是否回退到文本模式
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.fallback_to_text = fallback_to_text
        self.available = False
        self.device = "cpu"

        # 检查依赖是否可用
        if not TRANSFORMERS_AVAILABLE:
            if fallback_to_text:
                logger.warning("⚠️ transformers或torch未安装")
                logger.info("💡 将回退到文本模式")
                return
            else:
                raise RuntimeError("transformers或torch未安装，且fallback_to_text=False")

        # 检测设备
        assert torch is not None  # type: ignore
        if device is None:
            if torch.cuda.is_available():  # type: ignore
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # 加载模型
        try:
            self._load_model()
            self.available = True
            logger.info(f"✅ 视觉编码器初始化成功: {self.model_name} ({self.device})")
        except Exception as e:
            if fallback_to_text:
                logger.warning(f"⚠️ 视觉编码器加载失败: {e}")
                logger.info("💡 将回退到文本模式")
                self.available = False
            else:
                raise RuntimeError(f"视觉编码器加载失败: {e}")

    def _load_model(self):
        """加载模型和分词器"""
        assert AutoModel is not None  # type: ignore
        assert AutoProcessor is not None  # type: ignore

        model_path = self.MODELS.get(self.model_name, self.model_name)

        logger.info(f"加载 SigLIP 模型: {model_path} (设备: {self.device})")

        # 加载模型和处理器
        self.model = AutoModel.from_pretrained(  # type: ignore
            model_path,
            cache_dir=self.cache_dir
        ).to(self.device).eval()

        # SigLIP不需要tokenizer，直接使用图像处理器
        self.processor = AutoProcessor.from_pretrained(  # type: ignore
            model_path,
            cache_dir=self.cache_dir
        )

        self.embedding_dim = self.model.config.vision_config.hidden_size
        logger.info(f"模型加载成功，嵌入维度: {self.embedding_dim}")

    def encode_image(self,
                     image: Union[Image.Image, str, bytes],
                     normalize: bool = True) -> Optional[List[float]]:
        """
        对图像进行编码

        Args:
            image: PIL Image对象，文件路径，或图像字节
            normalize: 是否归一化向量

        Returns:
            图像嵌入向量，如果不可用则返回 None
        """
        if not self.available:
            return None
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(f"不支持的图像类型: {type(image)}")

        # 预处理
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # 编码
        assert torch is not None
        with torch.no_grad():
            vision_outputs = self.model.vision_model(**inputs)
            image_embeds = vision_outputs.pooler_output

        # 转换为列表
        embedding = image_embeds.cpu().numpy()[0].tolist()

        # 归一化（可选）
        if normalize:
            import numpy as np
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()

        return embedding

    def encode_images(self,
                      images: List[Union[Image.Image, str, bytes]],
                      batch_size: int = 8,
                      normalize: bool = True) -> List[Optional[List[float]]]:
        """
        批量编码图像

        Args:
            images: 图像列表
            batch_size: 批处理大小
            normalize: 是否归一化向量

        Returns:
            图像嵌入向量列表，不可用的返回 None
        """
        if not self.available:
            return [None] * len(images)

        embeddings = []

        try:
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]

                # 加载并预处理图像
                pil_images = []
                for img in batch:
                    if isinstance(img, str):
                        img = Image.open(img).convert("RGB")
                    elif isinstance(img, bytes):
                        img = Image.open(io.BytesIO(img)).convert("RGB")
                    pil_images.append(img)

                # 批处理
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

                assert torch is not None
                with torch.no_grad():
                    vision_outputs = self.model.vision_model(**inputs)
                    image_embeds = vision_outputs.pooler_output

                # 转换为列表
                batch_embeddings = image_embeds.cpu().numpy()

                for emb in batch_embeddings:
                    if normalize:
                        norm = float((emb ** 2).sum() ** 0.5)
                        if norm > 0:
                            emb = emb / norm
                    embeddings.append(emb.tolist())

            return embeddings

        except Exception as e:
            logger.error(f"❌ 批量编码失败: {e}")
            return [None] * len(images)

    def encode_text(self, text: str, normalize: bool = True) -> Optional[List[float]]:
        """
        对文本进行编码（可选功能）

        Args:
            text: 文本
            normalize: 是否归一化

        Returns:
            文本嵌入向量，如果不可用则返回 None
        """
        if not self.available:
            return None

        try:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

            assert torch is not None
            with torch.no_grad():
                text_outputs = self.model.text_model(**inputs)
                text_embeds = text_outputs.pooler_output

            embedding = text_embeds.cpu().numpy()[0].tolist()

            if normalize:
                import numpy as np
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()

            return embedding

        except Exception as e:
            logger.error(f"❌ 文本编码失败: {e}")
            return None

    @property
    def dimension(self) -> Optional[int]:
        """返回嵌入维度，如果不可用则返回 None"""
        if not self.available:
            return None
        return getattr(self, 'embedding_dim', None)

    @property
    def is_available(self) -> bool:
        """检查视觉编码器是否可用"""
        return self.available


class SigLIPMultiModalEncoder(VisionEncoder):
    """SigLIP 多模态编码器（支持图像和文本联合编码）"""

    def encode_image_text_pair(self,
                                image: Union[Image.Image, str, bytes],
                                text: str,
                                normalize: bool = True) -> Optional[List[float]]:
        """
        联合编码图像-文本对（用于图表检索）

        Args:
            image: 图像
            text: 配对文本（如标题、说明）
            normalize: 是否归一化

        Returns:
            联合嵌入向量，如果不可用则返回 None
        """
        if not self.available:
            return None

        try:
            # 加载图像
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, bytes):
                image = Image.open(io.BytesIO(image)).convert("RGB")

            # 联合编码
            inputs = self.processor(
                images=image,
                text=text,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            assert torch is not None
            with torch.no_grad():
                outputs = self.model(**inputs)
                combined_embeds = outputs.image_embeds + outputs.text_embeds

            embedding = combined_embeds.cpu().numpy()[0].tolist()

            if normalize:
                import numpy as np
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (np.array(embedding) / norm).tolist()

            return embedding

        except Exception as e:
            logger.error(f"❌ 联合编码失败: {e}")
            return None


# 便捷函数
def create_vision_encoder(model_name: str = "siglip-so400m",
                          device: str = "auto",
                          cache_dir: Optional[str] = None,
                          fallback_to_text: bool = True) -> VisionEncoder:
    """
    创建视觉编码器的便捷函数

    Args:
        model_name: 模型名称
        device: 设备 (auto/cpu/cuda)
        cache_dir: 缓存目录
        fallback_to_text: 如果不可用是否返回降级实例

    Returns:
        VisionEncoder 实例（可能不可用）
    """
    return VisionEncoder(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
        fallback_to_text=fallback_to_text
    )
