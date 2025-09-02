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
- 自动生成随机种子确保图像多样性
- 智能参数处理机制

## 配置项说明

插件支持以下配置项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| invoke_url_base | https://ai.api.nvidia.com/v1/genai/ | API 基础 URL，用于拼接模型名称 |
| model | stabilityai/stable-diffusion-3-medium | 要使用的生成模型 |
| mode | "" | 模式参数，若为空则在提交时移除 |
| is_reference_diagram | false | 是否使用参考图片作为输入（不建议启用） |
| api_key | "" | 请求头 Authorization 所需的密钥 |
| cfg_scale | 5.0 | 图像与提示词的符合程度，小于0.1时将被移除 |
| steps | 50 | 图像生成的迭代次数，小于0.1时将被移除 |
| width | -1 | 生成图像的宽度，小于等于0时将被移除 |
| height | -1 | 生成图像的高度，小于等于0时将被移除 |
| aspect_ratio | 16:9 | 生成图像的宽高比，设置为"remove"时将被移除 |
| negative_prompt | "" | 不希望出现在图像中的内容，为空时将被移除 |
| timeout | 360 | API 请求的超时时间（秒） |

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

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](./LICENSE) 文件。