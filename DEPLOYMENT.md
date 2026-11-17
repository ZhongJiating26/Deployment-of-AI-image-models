# Windows 部署与验收

## 1. 复制项目并定位现有 Python

将项目文件复制到目标机的 `D:\AI\QwenImageAPI`。以下命令在目标机 PowerShell 中执行。若已验证的虚拟环境位于其他目录，把 `$py` 改成该环境下的 `Scripts\python.exe`。直接使用目标机上原有的环境。

```powershell
Set-Location D:\AI\QwenImageAPI
$py = 'D:\AI\QwenImageAPI\.venv\Scripts\python.exe'
& $py --version
& $py -c "import torch; from diffusers import QwenImage21Pipeline; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('Pipeline OK')"
& $py -m pip freeze | Set-Content -Encoding utf8 requirements.lock.txt
& $py -c "import fastapi, uvicorn, pydantic_settings, python_multipart, PIL; print('API dependencies OK')"
```

如果 API 层有缺包，先检查安装计划，再按需安装缺失的 Web 包。下例列出 Web 层依赖，命令省略 `--upgrade` 并用现有包清单约束版本。如果发生版本冲突，先核对目标机已有版本及冲突来源。

```powershell
$apiDeps = @('fastapi>=0.115,<1', 'uvicorn>=0.30,<1', 'pydantic>=2.10,<3', 'pydantic-settings>=2.6,<3', 'python-multipart>=0.0.20', 'python-dotenv>=1,<2')
& $py -m pip install --dry-run -c requirements.lock.txt $apiDeps
& $py -m pip install -c requirements.lock.txt $apiDeps
& $py -m pip check
```

`requirements.txt` 保留原始模板，包含推理依赖和客户端依赖。仅在重建环境时参照 README 的完整安装流程。OpenAI SDK 与替身测试也可在独立 Python 环境运行。若旧版本 Starlette 的 TestClient 提示缺少 HTTP 客户端，在测试环境安装 `httpx`。

## 2. 配置并启动

```powershell
Copy-Item .env.example .env
& $py -c "import secrets; print(secrets.token_urlsafe(48))"
notepad .env
```

将生成的密钥写入 `QWEN_API_KEY`。核对 `MODEL_PATH` 指向目标机已有模型根目录，其中包含 `model_index.json` 和完整权重。核对日志、上传和输出路径的写权限。保留默认 BF16、CPU offload 和 KV cache，先以默认分辨率验收。系统环境变量优先于项目根目录的 `.env`。相对目录以项目根目录解析。

```powershell
.\run.ps1 -PythonPath $py
```

如果 PowerShell 执行策略阻止脚本，可以直接运行 `& $py -m app.serve`。启动入口读取 `HOST` 与 `PORT`，固定单 worker。模型初始化完成后监听端口。模型加载失败时 `/health` 仍可检查进程，`/ready` 返回 503 `model_load_failed`，详情查询 `LOG_DIR\service.log`。

## 3. 运行验收

另开 PowerShell，设置相同的项目目录和 `$py`。安装了 OpenAI SDK 的客户端环境可运行以下命令。参数替换成实际 PNG、JPEG 或 WebP 文件。

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
& $py -m tests.smoke_live 'D:\AI\reference.png' --size 1024x1024
& $py -m unittest discover -s tests -v
```

真实验收会请求模型列表、文生图和单图编辑，在 `smoke-results` 写出两张校验过格式和尺寸的 PNG。打开图片确认文生图内容及编辑效果，再检查两条 `inference_finished` 日志的耗时和显存峰值。可使用 `--url http://127.0.0.1:9000` 指定自定义地址。替身测试无需 GPU、模型或服务进程。

## 4. 配置行为与故障排查

- `ENABLE_DOCS=true` 启用 `/docs` 和 `/openapi.json`。所有 `/v1` 请求携带 Bearer 密钥。使用 `CORS_ALLOWED_ORIGINS` 配置逗号分隔的浏览器来源。
- 已部署的 `.env` 会覆盖程序默认尺寸。启用横竖图时，将新版 `.env.example` 的 `ALLOWED_IMAGE_SIZES` 一行合并进去并重启。也可以添加其他宽高均在 512 至 2048 且为 32 倍数的尺寸。开启 `ENABLE_2048` 后自动加入 2048×2048，并启用配置列表中任一边为 2048 的尺寸。请在目标机验证显存用量。
- `MODEL_LOCAL_FILES_ONLY=true`、`MAX_IMAGES_PER_REQUEST=1`、`MAX_CONCURRENT_INFERENCES=1`、`ENABLE_URL_RESPONSE=false` 和 `DEFAULT_RESPONSE_FORMAT=b64_json` 是当前服务约束。启动时校验配置值。
- `MAX_QUEUE_LENGTH` 统计等待推理的请求。队满和超过 `QUEUE_WAIT_TIMEOUT_SECONDS` 均返回 429。取得推理槽后开始计算 `REQUEST_TIMEOUT_SECONDS`。超过该限时返回 504，线程继续持有推理槽直到本次任务结束。客户端及反向代理的响应超时宜覆盖排队与推理时限之和，并预留上传和传输时间。
- `SAVE_OUTPUTS` 与 `SAVE_UPLOADS` 分别控制输出 PNG 和原始参考图是否写盘。按 UTC 日期和请求 ID 存放。启动和运行期间每小时依据 `OUTPUT_RETENTION_DAYS` 清理当前启用目录中的过期服务图像。
- 日志按 UTC 日轮转，启动和运行期间每小时依据 `LOG_RETENTION_DAYS` 清理过期归档。默认记录 JSON，`LOG_PROMPTS=true` 可开启 prompt 记录。请求日志和推理日志用 `request_id` 关联，`peak_allocated_bytes` 是 PyTorch 已分配显存峰值。
- 503 时检查本地模型路径、Diffusers 类是否可导入、CUDA 和 CPU offload 依赖。500 `cuda_out_of_memory` 时检查其他 GPU 进程和请求尺寸。若恢复失败或 Pipeline 运行状态异常导致 `/ready` 返回 503，排查日志后重启服务。

## 5. 启用三个模型并按请求切换

先合并新版 `.env.example` 中新增的配置到现有 `.env`，保留原有密钥和目录。以下为路径示例，请换成已下载文件的实际位置。

```dotenv
MODEL_PATH=E:\model\qwen-image-2.1-turbo
QWEN_STANDARD_PATH=E:\model\qwen-image-2.1
MINIMAX_ENTROPACK_PATH=E:\model\MiniMax-H3-Image-EntroPack
MINIMAX_TEXT_ENCODER_PATH=E:\model\EntroPackPreQuants
MINIMAX_IMAGE_ASSETS_PATH=E:\model\MiniMax-H3-Image
MINIMAX_PROFILE=int8
QWEN_EDIT_REFERENCE_RESOLUTION=1024
```

`QWEN_STANDARD_PATH` 指向包含 `model_index.json` 的 Diffusers 标准版根目录。标准版运行 40 步，Turbo 沿用本地权重的 8 步采样计划。`QWEN_EDIT_REFERENCE_RESOLUTION=1024` 将两个 Qwen 模型的参考图处理分辨率上限设为 1024，输出仍由请求的 `size` 决定。请求 2048 输出需要 `ENABLE_2048=true`。

### MiniMax 所需文件

三个 MiniMax 路径一起填写。目录内部保留官方文件布局。选择与手头文件对应的配置。

| 本地根目录 | `int8` 所需文件 | `3bpp` 所需文件 |
| --- | --- | --- |
| `MINIMAX_ENTROPACK_PATH` | `models/dit_8bit_int8.safetensors`、`models/dit_8bit_int8.json`、`models/vae_int8.safetensors` | `models/dit_extreme_3bpp.safetensors`、`models/dit_extreme_3bpp.json`、`models/vae_int8.safetensors` |
| `MINIMAX_TEXT_ENCODER_PATH` | `models/MiniMax-H3/text_encoder_int8_convrot.safetensors` 和同名 JSON | `models/MiniMax-H3/text_encoder_3bpp.safetensors` 和同名 JSON |
| `MINIMAX_IMAGE_ASSETS_PATH` | `pdd_fused_heads.safetensors`、完整的 `processor/` | 同左 |

默认 `int8` 使用 CPU offload 和 20 GiB 的 DiffSynth VRAM 管理目标，适合有较多系统内存的机器。该目标用于模型权重管理，推理总峰值仍应实测。`3bpp` 使用配套的混合低比特权重，按官方消费卡配方关闭模型 offload。[DiffSynth 官方 24 GB 配方](https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/MiniMax-H3-Image.html)

如果已有 `dit_7bit_int8.safetensors`，选择 `MINIMAX_PROFILE=7bit-int8`。此配置从 EntroPack 的 `models/` 读取 `dit_7bit_int8.safetensors`、`vae_int8.safetensors` 和两者的同名 JSON，并使用 CPU offload。文本编码器目录须包含 `models/MiniMax-H3/text_encoder_3bpp.safetensors` 和同名 JSON。图像资源目录同样需要 `pdd_fused_heads.safetensors` 和完整的 `processor/`。INT8 GEMM 路径要求可正常运行的 Triton CUDA 后端，Windows 目标机需确认其与现有 Python、PyTorch 和 CUDA 构建兼容。[EntroPack 权重说明](https://modelscope.ai/models/DiffSynth-Studio/MiniMax-H3-Image-EntroPack)

### Windows 可选依赖

两个 Qwen 模型继续复用已验证的 Python 环境。只有启用 MiniMax 才需 DiffSynth 2.2.2、EntroPack 及对应 CUDA 版本的 CuPy。下例对应 CUDA 12。先生成现有环境的版本约束，再检查安装计划。确认解析成功且仅添加缺失包后执行安装。如遇冲突或 Windows 缺少可用 wheel，保留原环境并先验证独立环境的兼容性。

```powershell
& $py -m pip freeze --exclude-editable | Where-Object { $_ -match '^[^ =]+==' } | Set-Content -Encoding utf8 requirements.lock.txt
$miniDeps = @('diffsynth==2.2.2', 'cupy-cuda12x')
& $py -m pip install --dry-run -c requirements.lock.txt $miniDeps
# 检查安装计划后，才执行下面的安装。
& $py -m pip install -c requirements.lock.txt $miniDeps
& $py -m pip check
& $py -c "import torch, cupy, entropack; from diffsynth.pipelines.minimax_h3_image import MiniMaxH3ImagePipeline; print(torch.__version__, torch.version.cuda, cupy.cuda.runtime.getDeviceCount())"
```

### 客户端选择与验收

重启后，`GET /v1/models` 列出已配置的模型 ID。文生图的 JSON 和图片编辑的 multipart 均使用同一个 `model` 字段。省略该字段时使用 Turbo 的 `MODEL_NAME`。先在 1024 输出完成三种模型的真实文生图和编辑验收。

```powershell
& $py -m tests.smoke_live 'D:\AI\reference.png' --model 'Qwen/Qwen-Image-2.1-Turbo' --results smoke-results\turbo
& $py -m tests.smoke_live 'D:\AI\reference.png' --model 'Qwen/Qwen-Image-2.1' --results smoke-results\standard
& $py -m tests.smoke_live 'D:\AI\reference.png' --model 'DiffSynth-Studio/MiniMax-H3-Image-EntroPack' --results smoke-results\minimax
```

启动时预加载 Turbo。请求其他模型时，服务在同一单卡队列内释放当前 Pipeline，再从本地加载目标模型。接着请求同一个模型会复用它。切换耗时计入 `REQUEST_TIMEOUT_SECONDS`，并写入 `model_loaded` 日志。若 MiniMax 文件缺失，对该模型的请求返回 503，缺件清单在 `minimax_assets_missing` 日志中。切回可用模型时服务会重新加载该模型。
