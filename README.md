# Qwen-Image-2.1-Turbo · OpenAI 兼容图像 API

本仓库包含 FastAPI 服务、Windows 启动脚本及接口测试。操作步骤与排查方法见 [Windows 部署与验收](DEPLOYMENT.md)。

Turbo 为默认模型。可在 `.env` 中启用 Qwen-Image-2.1 标准版和 MiniMax-H3-Image-EntroPack，再通过图像请求的 `model` 字段切换。单卡同一时间只保留一个 Pipeline。各模型所需的本地目录和验收命令见 [三模型配置](DEPLOYMENT.md#5-启用三个模型并按请求切换)。

## 1. 项目概述

将本地 **Qwen-Image-2.1-Turbo** 模型封装为仅供授权客户端调用的 HTTP 服务，提供 OpenAI Images API 兼容的文生图及图片编辑能力。

- 推理框架：Diffusers `QwenImage21Pipeline` + PyTorch
- Web 框架：FastAPI + Uvicorn
- 运行系统：Windows（原生）
- 显卡：NVIDIA RTX 5090 D v2，24GB
- Python：3.14.10（以已验证成功的现有环境为准）
- 项目目录：`D:\AI\QwenImageAPI`
- 本地模型目录：`E:\model\qwen-image-2.1-turbo`
- 模型标识：`Qwen/Qwen-Image-2.1-Turbo`

### 当前已验证的基线

现有环境已成功使用 `QwenImage21Pipeline`、BF16 及 `enable_model_cpu_offload()` 在 1024×1024 生成图片；实测生成时间约 **15.72 秒**，PyTorch 报告的**峰值已分配 GPU 显存**约 **16.39 GiB**。该数据是单次测试，不是服务 SLA，也不等同于设备总显存占用。

Turbo 模型自带 8 步采样计划；不要沿用标准版 40 步配置，也不应无根据地覆盖其 scheduler。模型推理使用 `use_kv_cache=True`。

## 2. 功能范围

| 端点 | 功能 | 备注 |
| --- | --- | --- |
| `GET /health` | 存活检查 | 只返回最少信息 |
| `GET /ready` | 就绪检查 | 模型加载完成、服务可接单时返回 200；否则 503 |
| `GET /v1/models` | 模型列表 | 需 Bearer API Key |
| `POST /v1/images/generations` | 文生图 | JSON 请求，`b64_json` 响应 |
| `POST /v1/images/edits` | 图片编辑 | `multipart/form-data`，Qwen 最多 10 张、MiniMax 最多 16 张参考图 |
| `GET /docs` | OpenAPI 调试页 | 仅在 `ENABLE_DOCS=true` 时启用 |

图像接口支持 `n=1`。默认尺寸覆盖 1:1、4:3、3:4、3:2、2:3、16:9、9:16。尺寸列表和默认值按模型配置。Qwen 两个模型单边最多 2048，任一边为 2048 时需打开 `ENABLE_2048`。MiniMax 默认还列出 `2048x2048`、`4096x4096`、`6272x2688` 和 `2688x6272`。未实现 `image URL` 返回、异步任务 ID、mask 和真正的多 GPU 并发。

**接口兼容性说明：** 本项目实现 OpenAI Images API 的常见入参与成功返回结构，不等于完整实现 OpenAI 的所有扩展参数。未知或暂不支持的参数应明确报错，不应静默忽略并造成误解。

`GET /v1/models` 中每个模型包含本地扩展字段 `supported_sizes`（允许请求的输出尺寸）和 `default_size`（请求省略 `size` 时的默认输出尺寸）。列表反映当前服务配置，实际显存需求取决于模型和任务。

## 3. 建议项目结构

```text
D:\AI\QwenImageAPI\
├── app\
│   ├── __init__.py
│   ├── main.py                  # 创建 App、lifespan、异常处理
│   ├── config.py                # pydantic-settings 读取 .env
│   ├── schemas.py               # 请求、响应及统一错误结构
│   ├── exceptions.py            # 业务异常与错误映射
│   ├── middleware.py            # Request ID、鉴权、计时
│   ├── logging_config.py        # 结构化日志、轮转
│   ├── model_service.py         # 单例模型加载、串行推理
│   ├── image_utils.py           # 文件校验、编码、保存
│   └── routes\
│       ├── health.py
│       ├── models.py
│       └── images.py
├── tests\
├── logs\
├── outputs\
├── uploads\
├── .env                         # 本机配置（严禁提交仓库）
├── .env.example                 # 无实际密钥的配置模板
├── requirements.txt
├── run.ps1
└── README.md
```

## 4. Windows 首次部署

在 PowerShell 中执行。当前机器**已经具备可正常生图的 Python/PyTorch/Diffusers 环境**时，不应重新创建 `.venv` 或盲目升级推理依赖。

```powershell
cd D:\AI\QwenImageAPI
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import torch; from diffusers import QwenImage21Pipeline; print(torch.__version__); print(torch.cuda.is_available()); print('Pipeline OK')"
```

若需要从零重建环境：

```powershell
cd D:\AI\QwenImageAPI
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 先安装适合 RTX 5090 D v2 的 CUDA 版 PyTorch（示例 CUDA 12.8）
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 再安装支持 Qwen-Image-2.1-Turbo 专用采样计划的 Diffusers
# 目标机 GitHub 不通时，可先在其他电脑下载官方源码 ZIP 后复制到目标机
.\.venv\Scripts\python.exe -m pip install "D:\AI\Downloads\diffusers-main"

# 最后安装 API 和其他运行依赖
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

> 上述 `D:\AI\Downloads\diffusers-main` 是**示例源码解压路径**，需要按实际位置修改。不要在已有、已验证运行的环境中直接覆盖安装。模型官方建议使用支持其采样计划的 Diffusers 源码版本；生产部署应固定源码提交版本或保留经验证的源码包及校验值。

复制配置模板，并编辑 `.env` 中的 API Key 等参数：

```powershell
Copy-Item .env.example .env
notepad .env
```

使用独立随机长密钥；不要直接使用 `.env.example` 中的占位符。模型路径必须存在并包含 `model_index.json`。

### 启动服务

```powershell
cd D:\AI\QwenImageAPI
.\run.ps1
```

仅在完成防火墙白名单、内网访问控制、鉴权和必要的反向代理/HTTPS 配置后，才将监听地址改成 `0.0.0.0`。切勿在 CUDA 单卡机器上启动多个 Uvicorn worker，避免每个进程重复加载模型。

### 健康及模型检查

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready

$headers = @{ Authorization = "Bearer <替换成 .env 中的 QWEN_API_KEY>" }
Invoke-RestMethod http://127.0.0.1:8000/v1/models -Headers $headers
```

## 5. API 请求规范

### 文生图：`POST /v1/images/generations`

请求：`Content-Type: application/json`、`Authorization: Bearer <key>`。

```json
{
  "model": "Qwen/Qwen-Image-2.1-Turbo",
  "prompt": "一把现代户外折叠露营椅，纯白背景，电商产品摄影",
  "size": "1024x1024",
  "n": 1,
  "seed": 42,
  "response_format": "b64_json"
}
```

成功响应（示意；`b64_json` 省略）：

```json
{
  "created": 1794890000,
  "data": [{"b64_json": "<PNG_BASE64>"}],
  "meta": {
    "request_id": "req_123456",
    "model": "Qwen/Qwen-Image-2.1-Turbo",
    "elapsed_seconds": 15.72,
    "seed": 42
  }
}
```

`created` 为实际服务器 Unix 时间戳；上述值只是示例。`meta` 属于本地扩展，不应改变 OpenAI SDK 预期的 `data[*].b64_json` 结构。不要在公开响应中返回 Windows 绝对路径。

### 图片编辑：`POST /v1/images/edits`

请求格式为 `multipart/form-data`，通过重复的 `image` 或 `image[]` 字段上传 PNG/JPEG/WebP 参考图。Qwen 最多 10 张，MiniMax 最多 16 张，服务按上传顺序传给模型。`MAX_IMAGE_PIXELS` 限制单次请求全部参考图的总像素数。`n=1` 表示只生成一张输出图。其他字段包括 `prompt`、`model`、`size`、`seed`、`response_format`。不支持的参数（如 `mask`）返回 400。

应用层不限制图像上传或输出的文件字节数。用户上传的参考图长边超过 `MAX_REFERENCE_EDGE`（默认 2048 像素）时，服务在推理前等比例缩小。参考图总像素上限按缩小后的尺寸累计。生成和编辑的输出保持请求的 `size`。

使用 Python SDK 时，向 `client.images.edit` 传入文件列表即可。示例复用下文的 `client` 配置。

```python
with open("subject.png", "rb") as subject, open("scene.png", "rb") as scene:
    result = client.images.edit(
        model="DiffSynth-Studio/MiniMax-H3-Image-EntroPack",
        image=[subject, scene],
        prompt="保留图 1 的人物、服装和姿势，将背景替换成图 2 的街景。",
        size="1024x1024",
    )
```

示例 Python 客户端（同时可用于验收文生图）：

```python
import os
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["QWEN_API_KEY"],
    timeout=600.0,
)

result = client.images.generate(
    model="Qwen/Qwen-Image-2.1-Turbo",
    prompt="一把户外折叠露营椅，纯白背景，产品摄影",
    size="1024x1024",
    n=1,
)

with open("api_result.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))
```

如果使用 `.env` 而非系统环境变量，请在测试脚本中通过 `python-dotenv` 加载 `.env`，或者启动客户端前设置 `$env:QWEN_API_KEY`。

## 6. 错误响应规范

所有应用层错误应返回 JSON 并携带 `X-Request-ID` 响应头，禁止向 API 客户端泄露 Python Traceback、API Key、本地绝对路径或原始异常中的敏感信息。服务器日志可保存必要的堆栈用于排查，但应脱敏。

```json
{
  "error": {
    "message": "Unsupported image size",
    "type": "invalid_request_error",
    "code": "unsupported_size",
    "request_id": "req_123456"
  }
}
```

| HTTP | `code` | 典型场景 |
| --- | --- | --- |
| 400 | `invalid_request` / `unsupported_model` / `unsupported_size` | 无效或不支持的参数 |
| 401 | `invalid_api_key` | 缺失或无效 Bearer Token |
| 404 | `not_found` | 未实现的接口 |
| 413 | `file_too_large` | 上传超过限制 |
| 415 | `unsupported_media_type` | 上传媒体格式不支持 |
| 422 | `validation_error` | 类型或字段校验失败 |
| 429 | `server_busy` | 等待队列已满或等待超时 |
| 500 | `image_generation_failed` / `image_edit_failed` | 推理失败 |
| 500 | `cuda_out_of_memory` | GPU 显存不足 |
| 503 | `model_not_ready` / `model_load_failed` | 模型不可用或加载失败 |
| 504 | `request_timeout` | 推理执行超过 HTTP 响应期限，运行中的任务仍占用单卡推理槽 |

其他未捕获异常统一转为 `internal_error`，保留相同结构，并记录 request_id；不要只依赖 FastAPI 默认的 `{"detail": ...}`。启动模型失败时应保持服务不可就绪、终止启动或明确返回 503，不应假报健康。

## 7. 可靠性与资源管理

- **单 Pipeline**：在 FastAPI lifespan 加载默认模型。相同模型的请求复用现有 Pipeline，切换模型时在推理槽内释放旧 Pipeline 后加载新模型。
- **串行推理**：同一时刻最多运行一个 GPU 推理任务；使用共享锁和有上限的等待队列，队列满返回 429。
- **请求线程**：耗时同步推理放在线程池/同步路由中，避免阻塞整个 ASGI 事件循环。
- **显存优化**：`dtype=torch.bfloat16` + `pipe.enable_model_cpu_offload()`，不要再对整个模型调用 `.to('cuda')`。
- **内存统计**：每次推理前调用 `torch.cuda.reset_peak_memory_stats()`，推理完成后记录 `torch.cuda.max_memory_allocated()`；这是 PyTorch 分配峰值，不是完整 GPU 显存总占用。
- **异常恢复**：CUDA OOM 记录异常并安全释放临时对象，必要时清理缓存；如果 GPU 状态不可恢复，应将服务标为不可用并重启，而不是无限重试。
- **请求超时**：配置客户端、反向代理与等待队列超时；不要假设 HTTP 断开能安全强制中断已经开始的 CUDA 推理。
- **文件安全**：限制上传字节数、真实解码格式、像素数量、防止图片解压炸弹；以服务端生成的 ID 命名输出，禁止客户端控制输出路径。
- **启动**：`--workers 1`；生产环境不要启用 `--reload`。

## 8. 配置、日志和数据保留

配置使用 `pydantic-settings` 从项目根目录 `.env` 读取，变量名以 `.env.example` 为准；所有配置项都应在代码里真实生效，未实现的配置不应对外宣称可用。敏感配置必须拒绝空值或占位符。

日志写入 `logs/`，按日期或文件大小轮转；包括 request_id、路径、状态码、队列等待时长、推理耗时、模型、尺寸、seed、PyTorch 显存峰值与错误码。默认**不记录完整 prompt、原图或 Bearer Token**。如需保存图像，按日期存储并执行保留期限清理；本地文件绝对路径仅供内部日志使用。

部署在企业内网前应设置入站防火墙规则、必要时使用反向代理 TLS，并只给可信客户端分发密钥。不要把 `/docs`、日志目录或输出目录直接开放给未授权访问者。

## 9. 验收清单

- [ ] `GET /health` 返回 200，`GET /ready` 正确反映加载状态
- [ ] 未带/错误 API Key 时，`/v1/models` 及生图接口返回 401
- [ ] 文生图返回 OpenAI SDK 可解析的 Base64 PNG
- [ ] 图片编辑接收 multipart 文件，返回 Base64 PNG
- [ ] 模型错误、非法尺寸、`n != 1`、不支持格式返回指定错误 JSON
- [ ] 上传超限返回 413，解码失败返回 400，过大像素图被拒绝
- [ ] 两个同时提交的请求不会触发两个并行 GPU 推理
- [ ] 队列已满返回 429，并记录可关联的 request_id
- [ ] GPU OOM 不泄露本地路径及堆栈到客户端
- [ ] 重启服务后模型只加载一次，且从本地目录读取
- [ ] 保留日志、输出清理、密钥脱敏等配置真实生效
- [ ] 在现有 RTX 5090 D v2 机器上验证正常生成与编辑

## 10. 企业合规提醒

Python、服务框架及模型权重遵循**不同的授权协议**。Qwen-Image-2.1-Turbo 当前模型仓库标注为 **Qwen Research License**；用于公司正式业务前，须由负责人员确认模型商业用途授权。应保留模型仓库版本、权重来源、第三方依赖清单、许可证和部署校验记录。此 README 不是法律意见。

## 11. 官方参考

- Qwen Turbo 模型页：<https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo>
- Qwen 项目：<https://github.com/QwenLM/Qwen-Image-2.1>
- Diffusers 官方仓库：<https://github.com/huggingface/diffusers>
- PyTorch CUDA 安装版本：<https://pytorch.org/get-started/previous-versions/>
- FastAPI 文档：<https://fastapi.tiangolo.com/>
