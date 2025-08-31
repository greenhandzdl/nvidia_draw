# NVIDIA Stable Diffusion 绘画插件

![License](https://img.shields.io/github/license/greenhandzdl/nvidia_sd_draw)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

基于 NVIDIA AI API 的图像生成插件，使用 Stable Diffusion 3 Medium 模型生成高质量图像。

## 功能介绍

本插件提供基于 NVIDIA AI API 的图像生成功能，通过调用 Stability AI 的 Stable Diffusion 3 Medium 接口生成图像，并返回 Base64 编码的图片数据，供后续对话或多模态处理使用。

插件包含三个主要函数：
- [nvidia_generate_image](./nvidia_stable_diffusion.py#L225-L291)：生成图像并返回Base64编码数据
- [nvidia_send_image](./nvidia_stable_diffusion.py#L292-L310)：发送图像数据
- [nvidia_draw](./nvidia_stable_diffusion.py#L311-L340)：主函数，整合图像生成和发送流程

主要特性：
- 集成 NVIDIA Stable Diffusion 3 Medium 模型
- 支持自定义提示词和图像比例
- 可配置负向提示词、采样步数和CFG Scale参数
- 支持模型组管理，可灵活切换API密钥
- 自动生成随机种子确保图像多样性

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

该插件作为一个Python模块，提供三个主要函数：

#### nvidia_draw (主函数)

```python
from nvidia_stable_diffusion import nvidia_draw

# 生成并发送图像
result = await nvidia_draw(
    prompt="一只在星空下奔跑的猫",
    aspect_ratio="16:9",
    _ctx=ctx  # Agent上下文
)

# 处理结果
if result["status"] == "success":
    # 图像生成并发送成功
    print(result["message"])
else:
    # 处理错误
    print(f"错误: {result['message']}")
```

#### nvidia_generate_image (仅生成图像)

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

#### nvidia_send_image (仅发送图像)

```python
from nvidia_stable_diffusion import nvidia_send_image

# 发送图像数据
result = await nvidia_send_image(image_base64_data, _ctx=ctx)

# 处理结果
if result["status"] == "success":
    print("图像发送成功")
else:
    print(f"图像发送失败: {result['message']}")
```

### 配置说明

插件支持以下配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| invoke_url_base | "https://ai.api.nvidia.com/v1/genai/" | API请求的基础URL |
| model | "stabilityai/stable-diffusion-3-medium" | 指定使用的生成模型 |
| api_key | 无 | API 鉴权密钥，用于请求头 "Authorization" |
| cfg_scale | 5.0 | 图像与prompt的符合程度，范围1.1-9 |
| steps | 50 | 图像生成的迭代次数，范围1-100 |
| aspect_ratio | "16:9" | 生成图像的宽高比，可选"1:1", "4:3", "3:4", "16:9", "9:16" |
| negative_prompt | "" | 负向提示词 |
| timeout | 360 | API请求的超时时间（秒） |

### 模型组配置

插件通过模型组管理API密钥，需要在系统配置中设置模型组：

1. 在系统配置中创建模型组
2. 设置模型组的API Key（从NVIDIA获取）
3. 在插件配置中指定使用的模型组名称

## API说明

### nvidia_draw

主函数，整合图像生成和发送流程。

参数：
- `prompt` (str): 正向提示词，描述希望生成的图像内容
- `aspect_ratio` (str): 目标宽高比，例如"16:9"、"1:1"
- `_ctx` (AgentCtx): Agent上下文

返回值：
- 成功时返回 `{"status": "success", "message": "Image generated and sent successfully"}`
- 失败时返回 `{"status": "error", "message": "<错误描述>"}`

### nvidia_generate_image

使用NVIDIA Stable Diffusion 3生成图像。

参数：
- `prompt` (str): 正向提示词，描述希望生成的图像内容
- `aspect_ratio` (str): 目标宽高比，例如"16:9"、"1:1"
- `_ctx` (AgentCtx): Agent上下文

返回值：
- 成功时返回 `{"image_base64": "<base64_string>"}`
- 失败时返回 `{"error": "<错误描述>"}`

### nvidia_send_image

发送Base64编码的图像数据。

参数：
- `image` (str): Base64编码的图像数据
- `_ctx` (AgentCtx): Agent上下文

返回值：
- 成功时返回 `{"status": "success", "message": "Image sent successfully"}`
- 失败时返回 `{"status": "error", "message": "Image sending failed: <错误描述>"}`

## 许可证

本项目采用MIT许可证，详情请见[LICENSE](./LICENSE)文件。

## 项目链接

- 项目地址: https://github.com/greenhandzdl/nvidia_sd_draw