# Qwen3-ASR 转写服务

基于 [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 的本地音视频转写 HTTP 服务。模型加载一次常驻显存，之后每次转写零加载开销——把音视频文件的路径（或文件本身）POST 过来，就拿到带时间戳的转写结果。

设计成一个**独立的转写基础设施**：不绑定任何具体项目，任何能发 HTTP 请求的东西——命令行、脚本、Web 前端、笔记工具、自动化流程、AI Agent——都可以直接调用。

> **模型文件不随仓库分发**（体积大、遵循模型自身许可），使用前需自行下载（约 4 GB）——见 **[第一步：下载模型](#第一步下载模型必做)**。

## 目录

- [特性](#特性)
- [环境要求](#环境要求)
- [**第一步：下载模型（必做）**](#第一步下载模型必做)
- [第二步：安装依赖](#第二步安装依赖)
- [第三步：配置并启动](#第三步配置并启动)
- [快速开始](#快速开始)
- [API 速览](#api-速览)
- [安全模型](#安全模型)
- [性能参考](#性能参考)
- [常见问题](#常见问题)
- [License](#license)

## 特性

- **模型常驻**：加载一次（约 4-5 秒），后续转写不再有模型加载等待
- **两种提交方式**：传本地路径（`/transcribe`）或直接上传文件（`/upload`），返回格式一致
- **Web 控制台**：浏览器打开首页即用，拖文件进来就能转，实时显示进度
- **自动排队**：并发请求自动串行排队，排队情况对所有人可见（`/status`）
- **跨进程 GPU 互斥**：通过文件锁与同机其他转写程序（如 whisper 类脚本）共享 GPU，不会打架
- **可中途取消**：按文件名终止运行中/排队中的任务（`/cancel`）
- **断连即止损**：调用方提前断开连接，转写在下一个切片边界自动放弃，不白烧 GPU
- **健壮性**：流式落盘防大文件 OOM、启动自动清理上次崩溃残留的临时文件
- **安全默认**：只绑定 `127.0.0.1`，并校验 Host 头防 DNS rebinding
- **30 种语言**：中/英/日/韩/德/法/西/俄等（见[官方模型卡](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)），支持中文方言

## 环境要求

| 组件 | 要求 |
|---|---|
| OS | Windows（实测）；Linux/macOS 理论可用（文件锁已做跨平台，欢迎反馈） |
| GPU | NVIDIA，约 4 GB 显存余量 |
| Python | 3.10+（实测 3.11） |
| ffmpeg | 在系统 PATH 中 |
| 磁盘 | 模型约 4 GB |

## 第一步：下载模型（必做）

模型文件体积大、遵循模型自身许可，**不随本仓库分发**，需自行从官方渠道下载（任选其一）：

| 渠道 | 地址 | 适合 |
|---|---|---|
| ModelScope | <https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B-hf> | 国内网络，速度快 |
| HuggingFace | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf> | 国际网络 |

**方式一：ModelScope CLI（国内推荐）**

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-ASR-1.7B-hf --local_dir ./models/Qwen3-ASR-1.7B-hf
```

**方式二：HuggingFace CLI**

```bash
pip install "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com   # 国内网络可走镜像，不需要则省略
huggingface-cli download Qwen/Qwen3-ASR-1.7B-hf --local-dir ./models/Qwen3-ASR-1.7B-hf
# 新版 huggingface_hub 也可用等价命令: hf download Qwen/Qwen3-ASR-1.7B-hf --local-dir ./models/Qwen3-ASR-1.7B-hf
```

**方式三：git 直接克隆**

```bash
git lfs install
git clone https://www.modelscope.cn/Qwen/Qwen3-ASR-1.7B-hf.git models/Qwen3-ASR-1.7B-hf
```

下载完成后目录里应至少有这些文件（约 4 GB）：

```
models/Qwen3-ASR-1.7B-hf/
├── config.json
├── model.safetensors      # 权重，约 3.8 GB
├── tokenizer.json
├── processor_config.json
├── chat_template.jinja
└── generation_config.json
```

> **注意**：必须用官方 `Qwen/Qwen3-ASR-1.7B-hf`。不要用第三方转存的 `eclipse005/Qwen3-ASR-1.7B`——其 config 是旧嵌套格式，与新版 transformers 不兼容，会导致加载死锁。

模型放在哪里都可以，启动时通过 `ASR_MODEL_DIR` 环境变量把路径告诉服务即可（见[第三步](#第三步配置并启动)）。

## 第二步：安装依赖

```bash
pip install -r requirements.txt
```

- Qwen3-ASR 自 **transformers 5.13.0** 起原生支持，装稳定版即可，无需 dev 版
- `torch` 建议按 [pytorch.org](https://pytorch.org) 给出的命令安装与本机 CUDA 匹配的 GPU 版

## 第三步：配置并启动

通过环境变量告诉服务模型在哪：

```bash
# Windows (CMD)
set ASR_MODEL_DIR=E:\path\to\Qwen3-ASR-1.7B-hf

# Linux / macOS
export ASR_MODEL_DIR=/path/to/Qwen3-ASR-1.7B-hf
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ASR_MODEL_DIR` | （必填） | 模型目录路径 |
| `ASR_DEVICE` | `cuda:0` | 推理设备 |
| `PORT` | `8003` | 监听端口（只绑定 127.0.0.1） |
| `ASR_LOCK_PATH` | 仓库根 `.transcribe.lock` | GPU 互斥锁文件；同机多个转写程序指向同一文件即可互相排斥 |

然后启动：

```bash
python asr_server.py        # 或 Windows 双击 start.bat / Linux ./start.sh
```

看到 `服务就绪,监听 :8003` 即可用（首次加载模型约 4-5 秒，期间 `/health` 返回 `loading`）。

## 快速开始

```bash
# 方式一：文件在本机，直接传路径
curl -X POST http://127.0.0.1:8003/transcribe \
  -H "Content-Type: application/json" \
  -d '{"audio_path": "E:/media/lecture.mp4"}'

# 方式二：直接上传文件
curl -X POST "http://127.0.0.1:8003/upload?language=Chinese" -F "file=@lecture.mp4"
```

返回：

```json
{
  "text": "完整转写文本……",
  "segments": [[0.0, 28.0, "第一段文本"], [28.0, 56.0, "第二段文本"]]
}
```

`segments` 每项为 `[开始秒, 结束秒, 该段文本]`，按 28 秒切片对齐，可直接转 SRT 字幕。

或者用 Python 示例客户端（自动保存 `.txt` 和 `.srt`）：

```bash
pip install requests
python examples/client.py E:/media/lecture.mp4 --out result
```

浏览器打开 `http://127.0.0.1:8003/` 还有内置 Web 控制台：选文件、选语言、转完自动下载 `.txt` / `.srt`，实时显示进度、排队和 GPU 状态。

## API 速览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web 控制台（HTML） |
| GET | `/health` | 健康检查（探活用） |
| GET | `/status` | 当前任务 / 排队队列 / GPU 状态 |
| POST | `/transcribe` | 传本地路径转写（同步返回） |
| POST | `/upload` | 上传文件转写（同步返回） |
| POST | `/cancel` | 终止运行中 / 排队中的任务 |
| GET | `/docs` · `/openapi.json` | 交互式 API 文档 / OpenAPI 规范 |

完整请求/响应格式、错误码表、排队与并发模型、各语言对接示例，见 **[docs/API.md](docs/API.md)**。

## 安全模型

- 只绑定 `127.0.0.1`，不监听局域网/公网地址
- 校验 Host 头，非 `localhost / 127.0.0.1 / ::1` 一律 `403`（防 DNS rebinding）
- **没有任何鉴权**：本机所有进程都能提交/取消任务、读到转写结果。请勿直接暴露到公网；如需远程访问，套带鉴权的反向代理（注意反代会改写 Host 头，需回写为 `127.0.0.1` 才能通过校验）
- `/transcribe` 接受任意本地路径，仅适合可信的本机/内网环境

## 性能参考

RTX 4060 Ti 16GB 实测，7 分 12 秒视频：

| 指标 | 数值 |
|---|---|
| 模型加载（启动时一次） | ~4.1s |
| 单文件转写 | ~57s（约 7.5 倍实时） |
| 显存占用（常驻） | ~3.8 GB |

长音频线性耗时：1 小时音频约 8 分钟量级（不含排队）。

## 常见问题

**Q：启动报"未配置模型目录"？**
没设 `ASR_MODEL_DIR` 环境变量，或路径不对。见"第一步：下载模型"。

**Q：`/health` 一直返回 `loading`？**
模型加载中，等 5-10 秒；超过一分钟看服务日志。

**Q：`/status` 里 `seg` 一直是 `0/0`？**
正常。前 6-10 秒在做 ffmpeg 提音和音频加载，之后进度开始走。

**Q：请求返回 499？**
任务被取消了——要么有人调了 `/cancel`，要么你的连接先断了（超时设短了、代理掐了连接）。加大客户端超时，或把 499 当作正常取消处理。

**Q：返回 500，detail 说"等待转写锁超时"？**
队列堵了约 30 分钟没轮到。先查 `/status` 看队列，或取消排在前面的任务。

**Q：转写文本的断句在切片边界有点怪？**
模型按固定 28 秒切片转写，段尾偶尔断句瑕疵属已知行为。要更自然的断句可减小 `CHUNK_SECONDS`（脚本头部），或改用 VAD 切分。

**Q：时间戳能精确到词吗？**
不能。`segments` 的时间戳粒度就是 28 秒切片边界，做字幕够用，做逐词对齐不行。

**Q：想转写其他语言？**
`language` 传对应语言名（`English`、`Japanese` 等，共 30 种）。控制台下拉框只列了中文/English，但 API 不受限。

## 致谢

- [Qwen3-ASR](https://huggingface.co/Qwen) — 阿里 Qwen 团队的语音识别模型（Apache-2.0）
- [FastAPI](https://fastapi.tiangolo.com/) / [Transformers](https://github.com/huggingface/transformers)

## License

[MIT](LICENSE)（模型文件遵循其自身许可，另行下载）
