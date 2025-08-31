创建名为 "nvidia_draw" 的插件，包含 "nvidia_generate_image" 和 "nvidia_send_image" 两个辅助函数，实现图像生成和发送流程。

1.  **用户可配置项：**

    *   **invoke_url_base:** 字符串类型，API请求的基础URL，用于拼接模型名称。默认值："https://ai.api.nvidia.com/v1/genai/"。
    *   **model:** 字符串类型，指定使用的生成模型。默认值："stabilityai/stable-diffusion-3-medium"。
    *   **api_key:** 字符串类型，API 鉴权密钥，用于请求头 "Authorization"。插件在 NGC 外部运行时必填。
    *   **cfg_scale:** 浮点数类型，图像与prompt的符合程度。有效范围：1.1 到 9，默认值：5。超出范围应报错。
    *   **steps:** 整数类型，图像生成的迭代次数。有效范围：1 到 100，默认值：50。超出范围应报错。
    *   **aspect_ratio:** 字符串类型，生成图像的宽高比。可选值："1:1", "4:3", "3:4", "16:9", "9:16"，默认值："16:9"。若输入不在可选值内，报错。
    *   **negative_prompt:** 字符串类型，描述不希望出现在图像中的内容。默认值：""。支持多行文本输入，配置项需声明`json_schema_extra=ExtraField(is_textarea=True).model_dump()`。
    *   **timeout:** 整数类型，API请求的超时时间，单位为秒。默认值：360。

2.  **Agent 配置项：**

    *   **prompt:** 字符串类型，描述希望生成的图像内容。由 Agent 动态配置，作为生成图像的关键指令。

3.  **随机生成项：**

    *   **seed:** 整数类型，控制图像生成的随机种子。范围：0 到 4294967295，插件自动生成。超出范围应报错。

4.  **nvidia_generate_image 函数：**

    *   构建 API 请求：
        *   拼接 `invoke_url`：`invoke_url = invoke_url_base + model`。
        *   构建请求头 `headers`：`headers = {"Authorization": "Bearer " + api_key}`。若`api_key`为空，且插件运行在 NGC 内部，则不添加此header。
        *   构建请求体 `payload`：包含字段 `prompt`，`cfg_scale`，`aspect_ratio`，`seed`，`steps`，`negative_prompt`，所有参数使用用户配置和Agent配置。
    *   使用 POST 方法向 `invoke_url` 发送请求，设置超时时间为用户配置的`timeout`秒。
    *   检查 HTTP 响应状态码。若状态码非 200，抛出异常，包含状态码和响应文本。
    *   从 JSON 响应体中提取 `image` 字段的值 (Base64 编码的图像数据)。

5.  **nvidia_send_image 函数：**

    *   接收 `image` 数据 (Base64 编码)。
    *   调用 Agent 提供的图片发送接口，发送 `image` 数据。指定图片类型为 "image/png"。
    *   若发送成功，返回 JSON 格式的成功消息：`{"status": "success", "message": "Image sent successfully"}`。
    *   若发送失败，捕获异常，返回 JSON 格式的错误消息：`{"status": "error", "message": "Image sending failed: [错误信息]"}`。

6.  **nvidia_draw 主函数：**

    *   调用 `nvidia_generate_image` 函数生成图像数据，捕获任何异常。
    *   若成功生成图像数据，调用 `nvidia_send_image` 函数发送图像数据，捕获任何异常。
    *   若 `nvidia_send_image` 函数发送成功，返回 JSON 格式的成功消息：`{"status": "success", "message": "Image generated and sent successfully"}`。
    *   若 `nvidia_generate_image` 或 `nvidia_send_image` 函数发生异常，捕获异常，返回 JSON 格式的错误消息：`{"status": "error", "message": "[错误信息]"}`。

7.  **异常处理：**

    *   `nvidia_generate_image` 请求失败时，返回 JSON 格式的错误消息，包含 HTTP 状态码和错误信息：`{"status": "error", "message": "Image generation failed: HTTP [状态码] - [错误信息]"}`。
    *   `nvidia_send_image` 发送图像失败时，返回 JSON 格式的错误消息，包含错误信息：`{"status": "error", "message": "Image sending failed: [错误信息]"}`。
    *   所有错误信息应包含足够的信息，方便用户调试。