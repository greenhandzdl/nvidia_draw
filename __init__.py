"""Nvidia Draw Plugin

基于 NVIDIA AI API 的图像生成插件，使用 Stable Diffusion 3 Medium 模型生成高质量图像。

主要功能:
- 图像生成：调用 NVIDIA API 生成图像
- Base64 编码图片数据返回：便于后续处理和传输
- 模型组管理：支持多API密钥切换

关键特性:
- 集成 NVIDIA Stable Diffusion 3 Medium 多种模型
- 支持自定义提示词和图像比例
- 可配置负向提示词、采样步数和CFG Scale参数
- 支持模型组管理，可灵活切换API密钥
- 自动生成随机种子确保图像多样性
"""
import base64
from importlib.util import source_hash
from pathlib import Path

import random
from typing import Any, Dict, Literal, Optional, Union

import aiofiles
import magic

import httpx
from pydantic import Field

from nekro_agent.core.core_utils import ConfigBase, ExtraField
from nekro_agent.tools.path_convertor import convert_to_host_path
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
# Plugin constants
# ----------------------------------------------------------------------
ABS_COMPARE: float = 0.1  # 用于比较浮点数是否接近0的阈值
EG_IMAGE: str = "data:image/png;example_id,0" # 默认参考图

# ----------------------------------------------------------------------
# Plugin instance
# ----------------------------------------------------------------------
plugin = NekroPlugin(
    name="nvidia_draw",
    module_name="nvidia_draw",
    description="适合于Nvidia供应的绘图插件。",
    version="0.3.1",
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
    mode: str = Field(
        default="",
        title="模式",
        description="在提交参数(mode)时，若为空，则将此参数移除。",
    )
    is_reference_diagram: bool = Field(
        default=False,
        title="是否使用参考图片",
        description="是否使用参考图片作为输入。(不建议启用：Nvidia API不让外部上传参考图像，都会触发422)",
    )
    api_key: str = Field(
        default="",
        title="API 鉴权密钥",
        description="请求头 Authorization 所需的密钥。",
    )
    cfg_scale: float = Field(
        default=5.0,
        title="CFG Scale",
        description="图像与提示词的符合程度，按照文档设置。若参数小于ABS_COMPARE，则在提交时将移除此参数。",
        ge=-1.0,
        le=9.0,
    )
    steps: int = Field(
        default=50,
        title="生成步数",
        description="图像生成的迭代次数，按照文档设置。若参数小于ABS_COMPARE，则在提交时将移除此参数。",
        ge=-1,
        le=300,
    )
    width: int = Field(
        default=-1,
        title="生成宽度",
        description="生成图像的宽度。若参数为0，则在提交时将移除此参数。",
        ge=-1,
        le=1024,
    )
    height: int = Field(
        default=-1,
        title="生成高度",
        description="生成图像的高度。若参数为0，则在提交时将移除此参数。",
        ge=-1,
        le=1024,
    )
    aspect_ratio: Literal[
        "1:1", "1:2", "2:1", "2:3", "3:2",
        "3:4", "4:3", "3:5", "5:3", "5:11",
        "6:7", "7:6", "7:9", "9:16", "16:9",
        "7:13", "9:21", "11:5", "13:7", "21:9",
        "match_input_image", "remove"
    ] = Field(
        default="16:9",
        title="宽高比",
        description="生成图像的宽高比。当设置为remove时，提交时将移除此参数。",
    )
    negative_prompt: str = Field(
        default="",
        title="负面提示词",
        description="不希望出现在图像中的内容。当为空时，提交时将移除此参数。",
        json_schema_extra=ExtraField(is_textarea=True).model_dump(),
    )
    timeout: int = Field(
        default=360,
        title="请求超时时间（秒）",
        description="API 请求的超时时间。",
        ge=1,
    )

# ----------------------------------------------------------------------
# 获取插件配置实例
# ----------------------------------------------------------------------
config = plugin.get_config(NvidiaDrawConfig)


# ----------------------------------------------------------------------
# 辅助函数：图像生成
# ----------------------------------------------------------------------
async def nvidia_generate_image(prompt: str,refer_image: str = EG_IMAGE) -> Union[bytes, Dict[str, str]]:
    """Generate an image using Nvidia's Stable Diffusion API.

    Args:
        prompt: The textual description of the desired image.
            Suggested elements to include:
            - Type of drawing (e.g., character setting, landscape, comics, etc.)
            - What to draw details (characters, animals, objects, etc.)
            - What they are doing or their state
            - The scene or environment
            - Overall mood or atmosphere
            - Very detailed description or story (optional, recommend for comics)
            - Art style (e.g., illustration, watercolor... any style you want)
        refer_image (str): Optional source image path for image reference (useful for image style transfer or keep the elements of the original image)

    Returns:
        On success, a Base64‑encoded PNG image bytes.
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
        "seed": seed,
    }

    # 条件性添加参数

    # 当mode不为remove时，添加此参数
    if config.mode:
        payload["mode"] = config.mode

    # 当is_reference_diagram为True时，添加此参数
    if config.is_reference_diagram:
        payload["image"] = refer_image

    # 当cfg_scale大于ABS_COMPARE时，才添加此参数
    if config.cfg_scale > ABS_COMPARE:
        payload["cfg_scale"] = config.cfg_scale

    # 当steps大于ABS_COMPARE时，才添加此参数
    if config.steps > ABS_COMPARE:
        payload["steps"] = config.steps

    # 当width大于0时，才添加此参数
    if config.width > 0:
        payload["width"] = config.width

    # 当height大于0时，才添加此参数
    if config.height > 0:
        payload["height"] = config.height

    # 当aspect_ratio为remove或负数时不添加此参数，否则添加
    if config.aspect_ratio != "remove":
        payload["aspect_ratio"] = config.aspect_ratio

    # 当negative_prompt非空时才添加此参数
    if config.negative_prompt:
        payload["negative_prompt"] = config.negative_prompt

    try:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.post(invoke_url, json=payload, headers=headers)
            # 检查 HTTP 状态码
            response.raise_for_status()
            data = response.json()

            image_str: Optional[str] = data.get("image") or data.get("artifacts")[0].get("base64")

            if not image_str:
                logger.error("Image generation failed: missing 'image' field in response")
                return {
                    "status": "error",
                    "message": "Image generation failed: Invalid response - missing 'image' field",
                }
            logger.debug("Image generation successful, size: %d bytes", len(image_str))

            image_bytes: bytes =  bytes(base64.b64decode(image_str))

            return image_bytes
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
async def nvidia_draw(_ctx: AgentCtx,
                      prompt: str,
                      refer_image: str = "",
                      send_to_chat: str = ""
                      ) -> Union[str, dict[str, str]]:
    """Generate an image from a prompt and send it to the user.

    Args:
        prompt: The textual description of the desired image.
            Suggested elements to include:
            - Type of drawing (e.g., character setting, landscape, comics, etc.)
            - What to draw details (characters, animals, objects, etc.)
            - What they are doing or their state
            - The scene or environment
            - Overall mood or atmosphere
            - Very detailed description or story (optional, recommend for comics)
            - Art style (e.g., illustration, watercolor... any style you want)
        refer_image (str): Optional source image path for image reference (useful for image style transfer or keep the elements of the original image)
        send_to_chat (str): if send_to_chat is not empty, the image will be sent to the chat_key after generation


    Returns:
        str: Generated image path.(Notice: This method returns a path to the generated image, not the image itself.)
            For example:``/app/uploads/sd_generate.jpeg``
            And you should use send_msg_file to send the image to the user.
        dict[str, str]: Error message if generation fails.
                        If you get this result, you should tell users that the image generation failed, the reason for the failure, and then retry after a while.
    Examples:
        prompt = "a illustration style cute orange cat napping on a sunny windowsill, watercolor painting style"

        # You should use send_msg_file to send the image to the user.
        # Generate new image and send to chat
        image_path = nvidia_draw(prompt)
        send_msg_file(image_path)

        # Modify existing image and send to chat
        image_path = nvidia_draw(prompt, "shared/refer_image.jpg")
        send_msg_file(image_path)

        # Or you can use the following method to send the image to the chat.(But I don't recommend this method)
        draw(prompt, "shared/refer_image.jpg", send_to_chat=_ck) # if adapter supports file, you can use this method to send the image to the chat. Otherwise, find another method to use the image.

        # Avoid Not Send Generated Image to chat
        # Generate new image but **NOT** send to chat
        nvidia_draw(prompt)

    """

    if refer_image:
        async with aiofiles.open(
            convert_to_host_path(Path(refer_image), chat_key=_ctx.chat_key, container_key=_ctx.container_key),
            mode="rb",
        ) as f:
            image_data = await f.read()
            mime_type = magic.from_buffer(image_data, mime=True)
            image_data = base64.b64encode(image_data).decode("utf-8")
        source_image_data :str = f"data:{mime_type};base64,{image_data}"
    else:
        source_image_data :str = EG_IMAGE

    # 生成图像
    gen_result = await nvidia_generate_image(prompt,source_image_data)
    if isinstance(gen_result, dict) and gen_result.get("status") == "error":
        error_msg: str = gen_result.get("message", "Unknown error")
        logger.error("Image generation error: %s", error_msg)
        logger.debug(error_msg)
        return {"status": "error", "message": error_msg}

    image_bytes: bytes = gen_result

    logger.debug("gen_result type: %s", type(image_bytes))
    logger.debug("Image generation successful, size: %d bytes", len(image_bytes))

    result_sandbox_file = await _ctx.fs.mixed_forward_file(image_bytes,"generate.jpeg")

    if send_to_chat:
        await _ctx.ms.send_image(send_to_chat, result_sandbox_file, ctx=_ctx)

    return result_sandbox_file


# ----------------------------------------------------------------------
# 清理方法
# ----------------------------------------------------------------------
@plugin.mount_cleanup_method()
async def clean_up() -> None:
    """Clean up any resources used by the plugin."""
    # 当前实现不需要额外的资源清理
    logger.info("nvidia_draw 插件资源已清理")
