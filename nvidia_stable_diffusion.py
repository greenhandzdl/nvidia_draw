"""NVIDIA Stable Diffusion 3 绘画插件

提供基于 NVIDIA AI API 的图像生成能力。插件通过调用
https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium
的 NVIDIA AI API 生成图像，并返回 Base64 编码的图片数据，供后续对话或多模态处理使用。

`CFG_SCALE` 与 `STEPS` 已迁移至插件配置，用户无需在调用时提供。  
API URL 也已迁移至插件配置，便于在不同部署环境下灵活修改。

配置中 `CFG_SCALE` 取值范围已更新为 **1.1 ~ 9**，`STEPS` 取值范围已更新为 **5 ~ 100**。
"""

from __future__ import annotations

import random
from typing import Any, Dict

import httpx
from pydantic import BaseModel, Field, logger
from pydantic import HttpUrl, validator

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.services.plugin.base import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)

# ----------------------------------------------------------------------
# 常量
# ----------------------------------------------------------------------
MAX_SEED: int = 2**31 - 1  # 32 位有符号整数的最大随机种子

# ----------------------------------------------------------------------
# 插件实例
# ----------------------------------------------------------------------
plugin = NekroPlugin(
    name="nvidia_stable_diffusion",
    module_name="nvidia_stable_diffusion",
    description="基于 NVIDIA Stable Diffusion 3 的绘画插件",
    version="0.1.0",
    author="greenhandzdl",
    url="https://github.com/greenhandzdl/nvidia_sd_draw",
)

# ----------------------------------------------------------------------
# 模型组配置模型（仅包含名称和 API Key）
# ----------------------------------------------------------------------
class ModelGroupConfig(BaseModel):
    """模型组的配置信息，仅包含名称和 API Key。

    Attributes
    ----------
    name: str
        模型组的唯一标识，供用户在插件配置中选择。
    api_key: str
        调用对应模型组的 NVIDIA AI API 所需的 Bearer Token。
    """

    name: str = Field(
        ...,
        title="模型组名称",
        description="用于在插件中选择的模型组标识。",
    )
    api_key: str = Field(
        ...,
        title="API Key",
        description="调用 NVIDIA Stable Diffusion 3 所需的 Bearer Token。",
    )

# ----------------------------------------------------------------------
# 插件配置
# ----------------------------------------------------------------------
@plugin.mount_config()
class NvidiaStableDiffusionConfig(ConfigBase):
    """插件运行时配置。

    Attributes
    ----------
    USE_DRAW_MODEL_GROUP: str
        系统中绘图模型组的名称。通过系统模型组配置获取对应的 API Key。
    TIMEOUT: int
        HTTP 请求超时时间（秒），防止网络卡顿导致插件长时间阻塞。
    NEGATIVE_PROMPT: str
        负向提示词，用于排除不希望出现的元素。
    STEPS: int
        采样步数，数值越大图像细节越丰富。取值范围 **5 ~ 100**，默认值由用户在插件配置中提供。
    CFG_SCALE: float
        Classifier‑Free Guidance Scale，控制生成图像的细节程度。取值范围 **1.1 ~ 9**，默认值由用户在插件配置中提供。
    API_URL: HttpUrl
        NVIDIA Stable Diffusion 3 的 API 端点。默认值指向官方示例地址，亦可在部署时自行覆盖。
    """

    USE_DRAW_MODEL_GROUP: str = Field(
        default="default-draw-chat",
        title="绘图模型组",
        json_schema_extra={
            "ref_model_groups": "True",
            "required": True,
            "model_type": "draw",
        },
        description="主要使用的绘图模型组，可在系统配置中对应模型组名称进行选择",
    )
    TIMEOUT: int = Field(
        default=30,
        title="请求超时时间（秒）",
        description="向 NVIDIA API 发起请求的最大等待时间。",
    )
    NEGATIVE_PROMPT: str = Field(
        default="",
        title="负向提示词",
        description="负向提示词，用于排除不希望出现的元素。",
        json_schema_extra=ExtraField(
            title="负向提示词",
            description="负向提示词，用于排除不希望出现的元素。",
        ),
    )
    STEPS: int
    CFG_SCALE: float
    API_URL: HttpUrl = Field(
        default="https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-3-medium",
        title="API URL",
        description="NVIDIA Stable Diffusion 3 的 API 端点地址，支持在插件配置中自定义。",
    )

    @validator("STEPS")
    def validate_steps(cls, v: int) -> int:
        """确保 steps 为 5 ~ 100 之间的整数。

        Args:
            v: 用户提供的 steps 值。

        Returns:
            验证后的 steps 值。

        Raises:
            ValueError: 当 steps 超出合法范围时抛出。
        """
        if not (5 <= v <= 100):
            raise ValueError("STEPS 必须在 5~100 之间")
        return v

    @validator("CFG_SCALE")
    def validate_cfg_scale(cls, v: float) -> float:
        """确保 cfg_scale 为 1.1 ~ 9 之间的验证。

        Args:
            v: 用户提供的 cfg_scale 值。

        Returns:
            验证后的 cfg_scale 值。

        Raises:
            ValueError: 当 cfg_scale 超出合法范围时抛出。
        """
        if not (1.1 <= v <= 9):
            raise ValueError("CFG_SCALE 必须在 1.1~9 之间")
        return v

    @validator("USE_DRAW_MODEL_GROUP")
    def validate_use_draw_model_group(cls, v: str) -> str:
        """确保绘图模型组名称非空。

        Args:
            v: 绘图模型组名称。

        Returns:
            验证后的模型组名称。

        Raises:
            ValueError: 当模型组名称为空字符串时抛出。
        """
        if not v.strip():
            raise ValueError("USE_DRAW_MODEL_GROUP 不能为空")
        return v


# ----------------------------------------------------------------------
# 获取插件配置实例（单例）
# ----------------------------------------------------------------------
config: NvidiaStableDiffusionConfig = plugin.get_config(NvidiaStableDiffusionConfig)


# ----------------------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------------------
def _error_response(message: str) -> Dict[str, str]:
    """记录错误日志并返回统一的错误响应结构。

    Args:
        message: 错误描述信息。

    Returns:
        包含单键 ``"error"`` 的字典。
    """
    logger.error(message)
    return {"error": message}


def _get_draw_model_group_config(name: str) -> ModelGroupConfig:
    """从系统配置中获取绘图模型组的 API Key。

    Args:
        name: 绘图模型组的名称。

    Returns:
        包含模型组名称和对应 API Key 的 ``ModelGroupConfig`` 实例。

    Raises:
        ValueError: 当系统未提供获取函数、模型组不存在或 API Key 缺失时抛出。
    """
    try:
        # 系统提供的函数用于获取模型组配置信息
        from nekro_agent.services.system_config import get_model_group_config
    except ImportError as exc:
        raise ValueError("系统未提供 `get_model_group_config` 方法") from exc

    try:
        # ``get_model_group_config`` 返回包含 ``api_key`` 键的字典
        group_cfg: Dict[str, Any] = get_model_group_config(name)  # type: ignore
        api_key: str | None: = group_cfg.get("api_key")
        if not api_key:
            raise ValueError("API key 缺失")
        return ModelGroupConfig(name=name, api_key=api_key)
    except Exception as exc:
        raise ValueError(f"获取模型组 '{name}' 配置失败: {exc}") from exc


# ----------------------------------------------------------------------
# 工具方法：生成图像
# ----------------------------------------------------------------------
@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="生成图像",
    description=(
        "使用 NVIDIA Stable Diffusion 3 生成图像并返回 Base64 编码的图片数据。"
        "`cfg_scale` 与 `steps` 参数已在插件配置中定义，无需在调用时提供。"
    ),
)
async def nvidia_generate_image(prompt: str, aspect_ratio: str, _ctx: AgentCtx) -> Dict[str, str]:
    """使用 NVIDIA Stable Diffusion 3 生成图像。

    根据用户提供的正向提示词和宽高比，调用插件配置中的
    ``CFG_SCALE`` 与 ``STEPS`` 参数向 NVIDIA AI API 发起请求。成功时返回
    ``{\"image_base64\": \"<base64_string>\"  }``；若出现错误则返回
    ``{\"error\": \"<错误描述>\"}``。

    Args:
        prompt: 正向提示词，描述希望生成的图像内容。
        aspect_ratio: 目标宽高比，例如 ``\"16:9\"``、``\"1:1\"``。

    Returns:
        成功时返回 ``{\"image_base64\": \"<base64_string>\"}``，错误时返回
        ``{\"error\": \"<错误描述>\"}``。

    Example:
        >>> nvidia_generate_image(
        ...     prompt="一只在星空下奔跑，颜色鲜艳",
        ...     aspect_ratio="16:9",
        ... )
    """
    # 1. 获取模型组的 API Key
    try:
        group_cfg: ModelGroupConfig = _get_draw_model_group_config(config.USE_DRAW_MODEL_GROUP)
    except ValueError as exc:
        return _error_response(str(exc))

    # 2. 生成随机种子，确保结果多样
    seed: int = random.randint(0, MAX_SEED)

    # 3. 构造请求头
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {group_cfg.api_key}",
        "Accept": "application/json",
    }

    # 4. 构造请求体
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "cfg_scale": config.CFG_SCALE,
        "aspect_ratio": aspect_ratio,
        "seed": seed,
        "steps": config.STEPS,
        "negative_prompt": config.NEGATIVE_PROMPT,
    }

    # 5. 发起请求并解析响应
    try:
        async with httpx.AsyncClient(timeout=config.TIMEOUT) as client:
            response = await client.post(str(config.API_URL), headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        return _error_response(f"NVIDIA Stable Diffusion 请求失败: {exc}")

    # 6. 提取图像 Base64 数据
    try:
        artifacts = data.get("artifacts")
        if not artifacts or not isinstance(artifacts, list):
            raise KeyError("artifacts")
        first_artifact = artifacts[0]
        image_base64 = first_artifact.get("base64")
        if not image_base64:  # type: ignore
            raise KeyError("base64")
        return {"image_base64": image_base64}
    except KeyError as exc:
        return _error_response(f"响应缺失预期字段: {exc}")
    except Exception as exc:
        logger.exception("生成图像时发生未知错误")
        return _error_response(f"未知错误: {exc}")


# ----------------------------------------------------------------------
# 清理资源
# ----------------------------------------------------------------------
@plugin.mount_cleanup_method()
async def clean_up() -> None:
    """清理插件资源。

    当前插件不持有长期连接或文件句柄，方法实现为空。若后续引入缓存或持久化资源，
    可在此处统一释放。
    """
    logger.info("NVIDIA Stable Diffusion 插件资源已清理")
