"""Nvidia Draw Plugin

基于 NVIDIA AI API 的图像生成插件，使用 Stable Diffusion 3 Medium 模型生成高质量图像。

主要功能:
- 图像生成：调用 NVIDIA API 生成图像
- Base64 编码图片数据返回：便于后续处理和传输
- 模型组管理：支持多API密钥切换

关键特性:
- 集成 NVIDIA Stable Diffusion 3 Medium 模型
- 支持自定义提示词和图像比例
- 可配置负向提示词、采样步数和CFG Scale参数
- 支持模型组管理，可灵活切换API密钥
- 自动生成随机种子确保图像多样性

插件包含的主要函数:
- [nvidia_generate_image](./__init__.py#L112-L180): 生成图像并返回Base64编码数据
- [nvidia_draw](./__init__.py#L184-L217): 主函数，整合图像生成和发送流程
- [clean_up](./__init__.py#L222-L228): 清理插件使用的资源
"""

import random
from typing import Any, Dict, Literal, Optional, Union
import base64

import httpx
from pydantic import Field, validator

from nekro_agent.core.core_utils import ConfigBase, ExtraField

from nekro_agent.api.schemas import AgentCtx
from nekro_agent.core import logger
from nekro_agent.api.plugin import (
    ConfigBase,
    ExtraField,
    NekroPlugin,
    SandboxMethodType,
)

import os

# ----------------------------------------------------------------------
# Plugin instance
# ----------------------------------------------------------------------
plugin = NekroPlugin(
    name="nvidia_sd_draw",
    module_name="nvidia_sd_draw",
    description="适合于Nvidia供应SD模型的绘图插件。",
    version="0.1.1",
    author="greenhandzdl",
    url="https://github.com/greenhandzdl/nvidia_sd_draw",
)


# ----------------------------------------------------------------------
# 配置类
# ----------------------------------------------------------------------
@plugin.mount_config()
class NvidiaDrawConfig(ConfigBase):
    """插件配置

    包含调用 Nvidia API 所需的参数。
    """

    invoke_url_base: str = Field(
        default="https://ai.api.nvidia.com/v1/genai/",
        title="API 基础 URL",
        description="用于拼接模型名称的基础 URL。",
    )
    model: str = Field(
        default="stabilityai/stable-diffusion-3-medium",
        title="模型名称",
        description="要使用的生成模型。",
    )
    api_key: str = Field(
        default="",
        title="API 鉴权密钥",
        description="请求头 Authorization 所需的密钥。若在 NGC 环境内部运行可留空。",
    )
    cfg_scale: float = Field(
        default=5.0,
        title="CFG Scale",
        description="图像与提示词的符合程度，范围 1.1~9。",
        ge=1.1,
        le=9.0,
    )
    steps: int = Field(
        default=50,
        title="生成步数",
        description="图像生成的迭代次数，范围 1~100。",
        ge=1,
        le=100,
    )
    aspect_ratio: Literal["1:1", "4:3", "3:4", "16:9", "9:16"] = Field(
        default="16:9",
        title="宽高比",
        description="生成图像的宽高比。",
    )
    negative_prompt: str = Field(
        default="",
        title="负面提示词",
        description="不希望出现在图像中的内容。",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    timeout: int = Field(
        default=360,
        title="请求超时时间（秒）",
        description="API 请求的超时时间。",
        ge=1,
    )

    @validator("cfg_scale")
    def validate_cfg_scale(cls, v: float) -> float:
        """确保 cfg_scale 在 1.1 到 9 之间。"""
        if not (1.1 <= v <= 9):
            raise ValueError("cfg_scale 必须在 1.1 到 9 之间")
        return v

    @validator("steps")
    def validate_steps(cls, v: int) -> int:
        """确保 steps 在 1 到 100 之间。"""
        if not (1 <= v <= 100):
            raise ValueError("steps 必须在 1 到 100 之间")
        return v


# ----------------------------------------------------------------------
# 获取插件配置实例
# ----------------------------------------------------------------------
config = plugin.get_config(NvidiaDrawConfig)


# ----------------------------------------------------------------------
# 辅助函数：图像生成
# ----------------------------------------------------------------------
async def nvidia_generate_image(prompt: str) -> Union[bytes, Dict[str, str]]:
    """Generate an image using Nvidia's Stable Diffusion API.

    Args:
        prompt: The textual description of the desired image.

    Returns:
        On success, a Base64‑encoded PNG image string.
        On failure, a dictionary with keys "status" and "message" describing the error.

    Raises:
        Any exception raised by the httpx library.
    """
    # 拼接完整的调用 URL
    invoke_url: str = f"{config.invoke_url_base}{config.model}"

    # 构建请求头，仅在提供了 API 密钥时添加 Authorization
    headers: Dict[str, str] = {}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    # 随机生成种子
    seed: int = random.randint(0, 4294967295)

    # 构建请求体
    payload: Dict[str, Any] = {
        "prompt": prompt,
        "cfg_scale": config.cfg_scale,
        "aspect_ratio": config.aspect_ratio,
        "seed": seed,
        "steps": config.steps,
        "negative_prompt": config.negative_prompt,
    }

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(invoke_url, json=payload, headers=headers)
            # 检查 HTTP 状态码
            response.raise_for_status()
            data = response.json()
            image_base64: bytes = data.get("image")
            if not image_base64:
                logger.error("Image generation failed: missing 'image' field in response")
                return {
                    "status": "error",
                    "message": "Image generation failed: Invalid response - missing 'image' field",
                }
            # 将Base64字符串解码为bytes
            logger.debug("Image generation successful, size: %d bytes", len(image_base64))
            return image_base64
    except httpx.HTTPStatusError as e:
        logger.error("Image generation HTTP error: %s", e)
        return {
            "status": "error",
            "message": f"Image generation failed: HTTP {e.response.status_code} - {e.response.text}",
        }
    except httpx.RequestError as e:
        logger.error("Image generation request error: %s", e)
        return {
            "status": "error",
            "message": f"Image generation failed: {str(e)}",
        }
    except Exception as e:
        logger.exception("Unexpected error during image generation")
        return {
            "status": "error",
            "message": f"Image generation failed: {str(e)}",
        }


# ----------------------------------------------------------------------
# 主方法：生成并发送图像
# ----------------------------------------------------------------------
@plugin.mount_sandbox_method(
    SandboxMethodType.TOOL,
    name="生成并发送图像",
    description="使用 Nvidia Stable Diffusion 生成图像并发送给用户。",
)
async def nvidia_draw(_ctx: AgentCtx, prompt: str) -> Union[str, dict[str, str]]:
    """Generate an image from a prompt and send it to the user.

    Args:
        prompt: The textual description of the desired image.

    Returns:
        success: str: The file path of the generated image.
        failure: dict[str, str]: A dictionary with keys "status" and "message" describing the error.

    Examples:
        # Generate new image but **NOT** send to chat
        nvidia_draw("a illustration style cute orange cat napping on a sunny windowsill, watercolor painting style")
    """
    # 生成图像
    gen_result = await nvidia_generate_image(prompt)
    if isinstance(gen_result, dict) and gen_result.get("status") == "error":
        error_msg: str = gen_result.get("message", "Unknown error")
        logger.error("Image generation error: %s", error_msg)
        return {"status": "error", "message": error_msg}

    image_bytes: bytes = gen_result  # type: ignore

    result_sandbox_file = await _ctx.fs.mixed_forward_file(image_bytes)
    return result_sandbox_file


# ----------------------------------------------------------------------
# 清理方法
# ----------------------------------------------------------------------
@plugin.mount_cleanup_method()
async def clean_up() -> None:
    """Clean up any resources used by the plugin."""
    # 当前实现不需要额外的资源清理
    logger.info("nvidia_draw 插件资源已清理")
