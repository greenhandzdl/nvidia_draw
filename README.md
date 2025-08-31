# NVIDIA Stable Diffusion 绘画插件

![License](https://img.shields.io/github/license/greenhandzdl/nvidia_sd_draw)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

基于 NVIDIA AI API 的图像生成插件，使用 Stable Diffusion 3 Medium 模型生成高质量图像。

## 功能介绍

本插件提供基于 NVIDIA AI API 的图像生成功能，通过调用 Stability AI 的 Stable Diffusion 3 Medium 接口生成图像，并返回 Base64 编码的图片数据，供后续对话或多模态处理使用。

主要特性：
- 集成 NVIDIA Stable Diffusion 3 Medium 模型
- 支持自定义提示词和图像比例
- 可配置负向提示词、采样步数和CFG Scale参数
- 支持模型组管理，可灵活切换API密钥

## 安装要求

- Python 3.8 或更高版本
- NVIDIA API Key (需在NVIDIA API Catalog中申请)

## 安装方法

1. 克隆本项目：
   ```bash
   git clone https://github.com/greenhandzdl/nvidia_sd_draw.git
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法

### 基本用法

该插件作为一个Python模块，主要提供一个异步函数[nvidia_generate_image](./nvidia_stable_diffusion.py#L225-L291)用于生成图像：

```python
from nvidia_stable_diffusion import nvidia_generate_image

# 生成图像
result = await nvidia_generate_image(
    prompt="一只在星空下奔跑的猫",
    aspect_ratio="16:9",
    _ctx=ctx  # Agent上下文
)

# 处理结果
if "image_base64" in result:
    image_data = result["image_base64"]
    # 处理图像数据
else:
    error_msg = result["error"]
    # 处理错误
```

### 配置说明

插件支持以下配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| USE_DRAW_MODEL_GROUP | "default-draw-chat" | 使用的绘图模型组名称 |
| TIMEOUT | 30 | HTTP请求超时时间（秒） |
| NEGATIVE_PROMPT | "" | 负向提示词 |
| STEPS | 50 | 采样步数（30-100） |
| CFG_SCALE | 7.5 | Classifier-Free Guidance Scale（0.5-15） |

### 模型组配置

插件通过模型组管理API密钥，需要在系统配置中设置模型组：

1. 在系统配置中创建模型组
2. 设置模型组的API Key（从NVIDIA获取）
3. 在插件配置中指定使用的模型组名称

## API说明

### nvidia_generate_image

使用NVIDIA Stable Diffusion 3生成图像。

参数：
- `prompt` (str): 正向提示词，描述希望生成的图像内容
- `aspect_ratio` (str): 目标宽高比，例如"16:9"、"1:1"
- `_ctx` (AgentCtx): Agent上下文

返回值：
- 成功时返回 `{"image_base64": "<base64_string>"}`
- 失败时返回 `{"error": "<错误描述>"}`

## 许可证

本项目采用MIT许可证，详情请见[LICENSE](./LICENSE)文件。

## 项目链接

- 项目地址: https://github.com/greenhandzdl/nvidia_sd_draw