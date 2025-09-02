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