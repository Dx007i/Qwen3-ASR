# API 参考

本文是 Qwen3-ASR 转写服务的完整接口说明与对接指南。安装与启动见主 [README](../README.md)。

服务是纯 HTTP + JSON，任何语言都能接。核心就两条路：**同机传路径**，**跨机传文件**。

---

## 接口一览

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web 控制台（HTML） |
| GET | `/health` | 健康检查（探活用） |
| GET | `/status` | 当前任务 / 排队队列 / GPU 状态 |
| POST | `/transcribe` | 传本地路径转写（同步返回） |
| POST | `/upload` | 上传文件转写（同步返回） |
| POST | `/cancel` | 终止运行中 / 排队中的任务 |
| GET | `/docs` · `/openapi.json` | 交互式 API 文档 / OpenAPI 规范 |
| GET | `/favicon.svg` | 站点图标 |

以下示例均假设服务跑在 `http://127.0.0.1:8003`。

### `GET /health` — 健康检查

轻量探活，不跑子进程，适合监控工具轮询。

```json
{"status": "ok", "model": "qwen3-asr-1.7b"}
```

模型加载中 `status` 为 `"loading"`，就绪后为 `"ok"`。

### `GET /status` — 状态查询

秒回。回答三个问题：**现在在转写什么、排了几个、GPU 忙不忙**。

```json
{
  "status": "busy",
  "current": {
    "audio_path": "E:/media/lecture.mp4",
    "language": "Chinese",
    "elapsed": 42.2,
    "seg": 10,
    "total_segs": 26
  },
  "queue_len": 1,
  "queue": [
    {"seq": 3, "audio_path": "E:/media/other.mp3", "language": "Chinese", "waiting": 14.1}
  ],
  "gpu": {"busy": true, "mem_gb": 3.8}
}
```

| 字段 | 说明 |
|---|---|
| `status` | `idle`（空闲）/ `busy`（有任务在转写） |
| `current` | 当前任务详情，空闲时为 `null` |
| `current.seg` / `total_segs` | 转写到第几段 / 共几段（28 秒一段） |
| `current.elapsed` | 当前任务已进行秒数 |
| `queue[].seq` | 请求序号（单调递增，即排队位置） |
| `queue[].waiting` | 该请求已等待秒数 |
| `gpu.busy` | GPU 是否被某个转写占用（文件锁非阻塞探测，含其他进程的转写） |
| `gpu.mem_gb` | 当前显存占用 |

> 提交前先查 `/status` 可以预估等待时间；多个调用方并发时，能在队列里看到彼此的请求。

### `POST /transcribe` — 路径转写

传**服务所在机器上的本地文件绝对路径**，同步等待转写完成后返回。

```json
{"audio_path": "E:/media/lecture.mp4", "language": "Chinese"}
```

- `language` 可选，默认 `Chinese`，可传 `English`、`Japanese` 等模型支持的 30 种语言
- 视频文件也可以，服务内部先用 ffmpeg 提取音轨
- ⚠️ 这是**同步长请求**：要等转写完成才返回，客户端 HTTP 超时务必设长（见"对接指南"）

**错误**：`400` 文件不存在；`500` 转写失败（`detail` 含原因）。

### `POST /upload` — 上传转写

`multipart/form-data` 上传文件，`language` 走 query 参数。与 `/transcribe` 返回格式完全一致。

```bash
curl -X POST "http://127.0.0.1:8003/upload?language=Chinese" -F "file=@lecture.mp4"
```

- 支持的扩展名：`.mp3 .wav .m4a .aac .flac .ogg .wma .mp4 .mkv .mov .avi .webm .flv`，其他类型返回 `400`
- 文件流式落盘（1 MB/块），再大的视频也不会把服务内存撑爆，转写完临时文件自动删除
- 排队、取消、断连放弃等行为与 `/transcribe` 完全相同
- 浏览器直接调用时：用户关闭/刷新页面，转写会在下一个切片边界自动放弃

**错误**：`400` 未提供文件或类型不在白名单；`500` 保存/转写失败。

### `POST /cancel` — 取消任务

终止运行中和排队中的任务。请求体可省略或为 `{}`：

```jsonc
// 不传 audio_path：终止当前正在运行的任务
{}

// 传 audio_path：按"子串匹配"终止，同时命中运行中 + 排队中的同名任务
{"audio_path": "lecture.mp4"}
```

成功返回：

```json
{
  "cancelled": true,
  "pattern": "lecture.mp4",
  "matched_running": true,
  "matched_pending": [{"seq": 3, "audio_path": "E:/media/lecture.mp4"}]
}
```

语义细节：

- **匹配是子串匹配**，传文件名即可命中完整路径；会同时终止所有匹配的运行中/排队中任务
- 排队中的任务立即移出队列；**运行中的任务在下一个 28 秒切片边界退出**（单段推理中途无法打断，取消延迟最多约一个切片的推理时间）
- 被取消的 `/transcribe` / `/upload` 请求收到 **`499`**
- 取消"上传的任务"时注意：`/upload` 任务的 `audio_path` 是服务的临时文件路径（`tmp*_upload.xxx`），不是你的原始文件名——请从 `/status` 的 `current.audio_path` / `queue[].audio_path` 里取真实路径来匹配

**错误**：`404` 当前无任务，或没有匹配到任何任务。

### `GET /docs`、`/openapi.json` — 机器可读的接口定义

服务是标准 FastAPI 应用，自带交互式文档（`/docs`）和 OpenAPI 3.1 规范（`/openapi.json`）。任何支持 OpenAPI 的工具（Postman、Apifox、代码生成器等）导入 `openapi.json` 即可生成对应语言的客户端。

---

## 统一返回与错误码

| 状态码 | 含义 | 典型场景 |
|---|---|---|
| `200` | 成功 | 返回 `{text, segments}` |
| `400` | 请求有误 | 路径文件不存在、上传类型不在白名单 |
| `404` | 无匹配 | `/cancel` 没找到可取消的任务 |
| `422` | 参数校验失败 | 缺 `audio_path` 字段等（FastAPI 标准） |
| `499` | 任务被终止 | 被 `/cancel` 取消，或调用方提前断开连接 |
| `500` | 服务端失败 | 转写出错、排队等锁超时（约 30 分钟上限） |

错误响应统一为 `{"detail": "中文错误说明"}`。**调用方应把 `499` 理解为"任务被取消"而非服务故障**，通常不应自动重试。

---

## 排队与并发模型

理解这几点，对接时才不会意外：

1. **严格串行**。GPU 推理同一时刻只跑一个任务。并发请求按到达顺序排队，先到先转。服务是单进程单 worker，模型只有一份。
2. **排队全程可见**。请求一进入处理流程（还没拿到 GPU 之前）就登记进队列，`/status` 能看到包括自己在内的所有等待者。等锁超时（约 1800 秒）的请求会被移出队列并返回 `500`，不会永久残留。
3. **跨进程 GPU 互斥**。服务通过跨进程文件锁（`asr_lock`）持锁转写。同机其他使用同一把锁的转写进程（例如 whisper 类脚本，把 `ASR_LOCK_PATH` 指向同一个锁文件即可）与本服务自动互斥，双方不会同时抢 GPU；`/status` 的 `gpu.busy` 也因此能反映真实的 GPU 占用。
4. **取消粒度 = 28 秒切片**。取消检查点在每段推理开始前，无法打断单段推理的中途。
5. **调用方断开 = 自动取消**。`/transcribe` 与 `/upload` 都会在切片边界检测客户端连接：调用方断开（超时被客户端掐掉、浏览器关页、进程被杀）后，任务放弃转写并释放 GPU，避免"结果没人要了还在白烧十几分钟"。**因此客户端超时不要设得过短**——设短了等于频繁取消任务。

---

## 对接指南

### Python（requests）

```python
import requests

ASR = "http://127.0.0.1:8003"

# 同机：传路径。timeout 给足(排队 + 转写)，或用 None 不限时
r = requests.post(f"{ASR}/transcribe", json={
    "audio_path": r"E:\media\lecture.mp4",
    "language": "Chinese",
}, timeout=None)
if r.status_code == 200:
    data = r.json()
    print(data["text"])            # 完整文本
    print(data["segments"][:2])    # [[0.0, 28.0, "..."], ...]
elif r.status_code == 499:
    print("任务被取消")
else:
    print("失败:", r.json()["detail"])
```

把 `segments` 转成 SRT 只需几行（完整可运行版本见 `examples/client.py`）：

```python
def to_srt(segments):
    ts = lambda s: f"{int(s//3600):02d}:{int(s%3600//60):02d}:{int(s%60):02d},{int(s*1000%1000):03d}"
    return "\n".join(
        f"{i}\n{ts(a)} --> {ts(b)}\n{t.strip()}\n"
        for i, (a, b, t) in enumerate(segments, 1) if t.strip())
```

### JavaScript / 浏览器（fetch）

```javascript
// 上传转写：文件来自用户选择
const fd = new FormData();
fd.append("file", fileInput.files[0]);
const r = await fetch("/upload?language=Chinese", { method: "POST", body: fd });
const data = await r.json();
// data.text → 纯文本；data.segments → [[start, end, text], ...]
```

> 浏览器场景有个天然福利：用户关掉页面，连接断开，服务端会自动放弃转写，不会留孤儿任务。

### Node.js / 其他后端

跨机或文件不在服务所在机器时用 `/upload`：

```javascript
import fs from "node:fs";

const buf = fs.readFileSync("lecture.mp4");
const r = await fetch("http://127.0.0.1:8003/upload?language=Chinese", {
  method: "POST",
  body: buf,
  headers: { "Content-Type": "audio/mpeg" },
});
const { text, segments } = await r.json();
```

### 命令行批处理

```bash
# 转完一个接一个：服务端天然排队，客户端挨个同步调用即可
for f in /media/*.mp4; do
  curl -s -X POST http://127.0.0.1:8003/transcribe \
    -H "Content-Type: application/json" \
    -d "{\"audio_path\": \"$f\"}" | jq -r .text > "${f%.*}.txt"
done
```

### 集成到既有系统

- **作为监控对象**：轮询 `GET /health` 探活，轮询 `GET /status` 采集队列长度/显存等指标
- **作为流程一环**：下载器/爬虫拿到媒体文件 → POST `/transcribe` → 拿 `text` 喂给后续 LLM 总结/归档
- **多客户端共存**：多个项目同时调用没问题，大家在 `/status` 队列里按序排队，互不干扰
- **想取消**：转写前记下 `audio_path`，需要中止时 POST `/cancel` 即可
- **任意语言客户端**：导入 `openapi.json` 自动生成
