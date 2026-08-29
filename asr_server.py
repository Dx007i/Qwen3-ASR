# -*- coding: utf-8 -*-
"""Qwen3-ASR-1.7B 常驻 HTTP 转写服务。

模型加载一次常驻显存,避免每次转写都重新加载模型(数十秒)。
转写请求走跨进程 GPU 互斥锁,单 worker 天然串行。

配置(环境变量,详见 README.md):
  ASR_MODEL_DIR  模型目录(必填,指向 Qwen/Qwen3-ASR-1.7B-hf 本地目录)
  ASR_DEVICE     推理设备(默认 cuda:0)
  PORT           监听端口(默认 8003,只绑定 127.0.0.1)
  ASR_LOCK_PATH  GPU 互斥锁文件路径(默认本目录下 .transcribe.lock;
                 同机多个转写程序想互相排斥,让它们指向同一个锁文件)

启动:python asr_server.py

接口:
  GET  /health       → {"status":"ok","model":"qwen3-asr-1.7b"}
  POST /transcribe   {"audio_path":"..."} → {"text":"...","segments":[[start,end,text],...]}
  POST /cancel       {"audio_path":"..."} → 按路径子串匹配终止运行中/排队中的任务
                     (不传 audio_path 则终止当前运行任务);被终止的转写请求返回 499
"""
import os
import sys
import time
import threading
import tempfile
import subprocess
from pathlib import Path

# GPU 防黑屏/防 TDR 配置
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

# 模型目录:Qwen/Qwen3-ASR-1.7B-hf 的本地目录(下载方式见 README.md),必填
MODEL_DIR = os.environ.get("ASR_MODEL_DIR", "")
DEVICE = os.environ.get("ASR_DEVICE", "cuda:0")
PORT = int(os.environ.get("PORT", "8003"))
CHUNK_SECONDS = 28  # 分片秒数

# 模块级单例(启动时加载一次,后续请求复用)
_processor = None
_model = None

# ── 任务状态追踪(供 /status 接口查询排队与进度) ──────────────────────
# 模块级单任务追踪 + 排队队列。
# _state_lock 保护下面三个变量(转写线程写,/status 读,读写都很短)。
_state_lock = threading.Lock()
_current_task = None   # {"audio_path","language","started_at","seg","total_segs"} 或 None
_pending = []          # [{"seq","audio_path","language","arrived_at"}] 按到达顺序
_seq_counter = 0
_cancel_requested = []  # 已登记取消的匹配串(/cancel 写入,转写检查点读)


class TranscribeCancelled(Exception):
    """转写被 /cancel 终止时抛出,HTTP 层统一转 499。"""


def _request_cancel(pattern: str) -> None:
    """登记取消匹配串(转写检查点用子串匹配 audio_path)。"""
    with _state_lock:
        if pattern not in _cancel_requested:
            _cancel_requested.append(pattern)


def _is_cancelled(audio_path: str) -> bool:
    """检查某任务的 audio_path 是否命中任一取消登记(子串匹配)。"""
    with _state_lock:
        return any(p in audio_path for p in _cancel_requested)


def _clear_cancel(audio_path: str) -> None:
    """任务结束时清理能命中它的取消登记,防止集合无限增长。"""
    with _state_lock:
        _cancel_requested[:] = [p for p in _cancel_requested if p not in audio_path]


def _next_seq() -> int:
    """生成单调递增的请求序号(=排队位置)。"""
    global _seq_counter
    with _state_lock:
        _seq_counter += 1
        return _seq_counter


def _register_pending(audio_path: str, language: str) -> int:
    """请求进入转写线程后第一件事:登记到排队队列,返回序号。

    在 acquire_transcribe_lock 之前调,这样等锁的请求也能被 /status 看到。
    """
    seq = _next_seq()
    with _state_lock:
        _pending.append({
            "seq": seq,
            "audio_path": audio_path,
            "language": language,
            "arrived_at": time.time(),
        })
    return seq


def _promote_to_running(seq: int, audio_path: str, language: str) -> None:
    """拿到 GPU 锁后、开始转写前:从队列移除,置为当前运行任务。"""
    global _current_task
    with _state_lock:
        _pending[:] = [p for p in _pending if p["seq"] != seq]
        _current_task = {
            "audio_path": audio_path,
            "language": language,
            "started_at": time.time(),
            "seg": 0,
            "total_segs": 0,
        }


def _update_progress(seg: int, total_segs: int) -> None:
    """分片循环里每段转写前更新进度(供 /status 看到第几段/共几段)。"""
    global _current_task
    with _state_lock:
        if _current_task is not None:
            _current_task["seg"] = seg
            _current_task["total_segs"] = total_segs


def _finish_current() -> None:
    """转写结束(无论成功失败)清空当前任务。"""
    global _current_task
    with _state_lock:
        _current_task = None


def _remove_pending(seq: int) -> None:
    """从排队队列移除指定序号的请求(等锁超时等异常路径调,避免条目残留)。"""
    with _state_lock:
        _pending[:] = [p for p in _pending if p["seq"] != seq]


def _probe_lock_held() -> bool:
    """非阻塞探测 asr_lock 文件锁是否被占用(=是否有转写在持锁占 GPU)。

    复用 asr_lock.LOCK_PATH,尝试非阻塞加锁:失败即被占用,成功立即释放,
    不真持锁。这样 /status 能准确反映 GPU 是否真忙,且自身不干扰转写。
    """
    try:
        import msvcrt

        def try_lock(fd):
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

        def unlock(fd):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except ImportError:
        try:
            import fcntl

            def try_lock(fd):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def unlock(fd):
                fcntl.flock(fd, fcntl.LOCK_UN)
        except ImportError:
            return False  # 两个平台的锁 API 都不可用,跳过探测
    from asr_lock import LOCK_PATH
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            try_lock(fd)
        except OSError:
            return True  # 被占用 → 有转写在跑
        try:
            unlock(fd)
        except OSError:
            pass
        return False
    finally:
        os.close(fd)


def extract_audio(src: str, dst: str) -> None:
    """用 ffmpeg 从视频/音频提取 16kHz 单声道 16-bit PCM wav。"""
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", dst,
    ]
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def cleanup_temp_files() -> None:
    """启动时清理上次崩溃/强杀残留的临时音视频文件。

    taskkill /F 跳过 do_transcribe 的 finally,会残留:
      tmp<随机>_16k.wav     (extract_audio 生成的临时 wav)
      tmp<随机>_upload.<ext> (/upload 保存的原始文件)
    用 os.O_EXCL 独占打开探测:打不开=被占用(正在用),跳过;能打开=残留,删除。
    """
    import glob
    patterns = [
        os.path.join(tempfile.gettempdir(), "tmp*_16k.wav"),
        os.path.join(tempfile.gettempdir(), "tmp*_upload.*"),
    ]
    removed = 0
    freed = 0
    for pat in patterns:
        for path in glob.glob(pat):
            # 独占打开探测是否被占用(Windows: O_EXCL 失败=已存在/被占用)
            # 用 _O_EXCL 不可靠(文件已存在就失败),改用尝试独占写:打开后立即删
            try:
                fd = os.open(path, os.O_RDWR)
            except OSError:
                # 被占用(进程正用)或无权限,跳过
                continue
            os.close(fd)
            try:
                size = os.path.getsize(path)
                os.remove(path)
                removed += 1
                freed += size
            except OSError:
                pass
    if removed:
        mb = freed / 1048576
        print(f"[asr-server] 清理上次残留临时文件 {removed} 个,释放 {mb:.1f}MB", flush=True)


def load_model():
    """加载 Qwen3-ASR processor 与模型(启动时调一次)。

    用 AutoModelForMultimodalLM(官方 README 推荐,不能用 AutoModelForSpeechSeq2Seq
    后者会 segfault)。low_cpu_mem_usage + device_map 直接流式加载到 GPU,
    不在 CPU 建完整副本。需要系统页面文件有足够余量(~4GB)。
    """
    if not MODEL_DIR:
        raise RuntimeError(
            "未配置模型目录:请设置环境变量 ASR_MODEL_DIR 指向 Qwen/Qwen3-ASR-1.7B-hf "
            "的本地模型目录(下载方式见 README.md)")
    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"模型目录不存在: {MODEL_DIR}(请检查 ASR_MODEL_DIR,下载方式见 README.md)")

    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    print(f"[asr-server] 加载 processor: {MODEL_DIR}", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    print(f"[asr-server] 加载模型 (low_cpu_mem_usage, device_map {DEVICE}, bfloat16)",
          flush=True)
    t0 = time.time()
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": DEVICE}, trust_remote_code=True,
    ).eval()
    device = next(model.parameters()).device
    gpu_gb = torch.cuda.memory_allocated(0) / 1073741824
    print(f"[asr-server] 模型加载耗时 {time.time()-t0:.1f}s, device={device}, "
          f"GPU占用={gpu_gb:.2f}GB", flush=True)
    return processor, model


def transcribe_audio(audio_path: str, processor, model, device: str,
                     language: str = "Chinese",
                     chunk_seconds: int = CHUNK_SECONDS,
                     cancel_check=None):
    """分片转写长音频,返回 (full_text, segments)。

    segments = [(start_sec, end_sec, text), ...]
    cancel_check: 每段开头调一次,返回 True 则抛 TranscribeCancelled 中止。
    """
    import numpy as np
    import torch
    import librosa

    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    total = len(y) / sr
    chunk_samples = chunk_seconds * sr
    n_chunks = int(np.ceil(len(y) / chunk_samples))

    print(f"[asr-server] 音频 {total:.1f}s, 分 {n_chunks} 段转写", flush=True)
    segments = []
    texts = []
    for i in range(n_chunks):
        # 段间取消检查点(generate 单段推理中途无法打断,粒度=段)
        if cancel_check is not None and cancel_check():
            raise TranscribeCancelled(f"任务已被终止: {audio_path}")
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(y))
        chunk = y[start:end]
        t_start, t_end = start / sr, end / sr
        if len(chunk) < sr * 0.1:
            continue

        inputs = processor.apply_transcription_request(
            audio=chunk, language=language, sampling_rate=sr,
        )
        dev_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                t = v.to(device)
                if v.dtype.is_floating_point:
                    t = t.to(torch.bfloat16)
                dev_inputs[k] = t
            else:
                dev_inputs[k] = v

        t0 = time.time()
        with torch.no_grad():
            generated = model.generate(**dev_inputs, max_new_tokens=2048)
        prompt_len = dev_inputs["input_ids"].shape[-1]
        new_ids = generated[0, prompt_len:]
        text = processor.decode(new_ids, skip_special_tokens=True,
                                return_format="transcription_only")
        dt = time.time() - t0
        print(f"[asr-server] seg {i+1}/{n_chunks} {t_start:.0f}-{t_end:.0f}s "
              f"({dt:.1f}s): {text[:70]}", flush=True)
        _update_progress(i + 1, n_chunks)
        segments.append((t_start, t_end, text))
        texts.append(text)

    return "".join(texts), segments


def do_transcribe(audio_path: str, language: str = "Chinese",
                  client_gone=None) -> dict:
    """完整转写流程:注册排队 → 持锁 → ffmpeg 提音 → 分片转写 → 清理。

    与其他使用同一把 asr_lock 锁文件的转写进程互斥,确保不同时占 GPU。
    被 /cancel 终止时抛 TranscribeCancelled(等锁阶段被取消则抛
    asr_lock.CancelledError),HTTP 层统一转 499。
    client_gone: 可选回调(无参,返回 bool),调用方(HTTP 客户端)断开时
    返回 True——转写结果已无人接收,在段间检查点放弃转写释放 GPU,
    避免孤儿任务白烧十几分钟还阻塞后续排队。
    """
    from asr_lock import (acquire_transcribe_lock, release_transcribe_lock,
                          CancelledError)

    if _model is None or _processor is None:
        raise RuntimeError("模型未加载")

    def _cancelled() -> bool:
        # /cancel 登记 或 调用方已断开
        if _is_cancelled(audio_path):
            return True
        return client_gone is not None and client_gone()

    # 先登记到排队队列(在等锁之前,这样 /status 能看到排在后面的请求)
    seq = _register_pending(audio_path, language)

    # 从模型参数取实际 device(device_map="auto" 可能不全是 cuda:0)
    import torch
    device = next(_model.parameters()).device

    try:
        # 排队阶段被取消:不用干等到拿锁,轮询中即可退出
        # (等锁阶段 client_gone 也参与检查:调用方断开后无需再排队)
        lock_fd = acquire_transcribe_lock(cancel_check=_cancelled)
    except CancelledError:
        # 等锁期间被 /cancel 终止:归一为 TranscribeCancelled(HTTP 499)
        _remove_pending(seq)
        _clear_cancel(audio_path)
        raise TranscribeCancelled(f"任务已被终止: {audio_path}")
    except Exception:
        # 等锁超时:从 _pending 移除本请求,避免条目永久残留导致 /status 虚报排队
        _remove_pending(seq)
        raise
    try:
        # 拿到锁后、开跑前再查一次(覆盖"等锁最后一轮与拿锁之间被取消"窗口)
        # 此时还没 _promote_to_running,需手动移除排队条目避免残留
        if _cancelled():
            _remove_pending(seq)
            raise TranscribeCancelled(f"任务已被终止: {audio_path}")
        _promote_to_running(seq, audio_path, language)
        # mkstemp 而非 mktemp:消除 TOCTOU 竞态
        _fd, wav_path = tempfile.mkstemp(suffix="_16k.wav")
        os.close(_fd)
        try:
            extract_audio(audio_path, wav_path)
            full_text, segments = transcribe_audio(
                wav_path, _processor, _model, device, language=language,
                cancel_check=_cancelled)
        except TranscribeCancelled:
            # 重抛时换成原始音频路径(对外报错不露临时 wav,方便调用方对账)
            raise TranscribeCancelled(f"任务已被终止: {audio_path}")
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
        return {"text": full_text,
                "segments": [[s, e, t] for s, e, t in segments]}
    finally:
        _finish_current()
        _clear_cancel(audio_path)
        release_transcribe_lock(lock_fd)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<title>Qwen3-ASR 转写控制台</title>
<style>
  :root { --bg:#0f1419; --card:#1a2029; --border:#2a3340; --txt:#e4e7eb;
          --dim:#8b95a5; --accent:#4a9eff; --green:#4ade80; --amber:#fbbf24; --red:#f87171; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:var(--txt); font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         padding:20px; max-width:1100px; margin:0 auto; }
  h1 { font-size:20px; font-weight:600; margin-bottom:16px; display:flex; align-items:center; gap:10px; }
  h1 .dot { width:10px; height:10px; border-radius:50%; background:var(--dim); }
  h1 .dot.busy { background:var(--green); box-shadow:0 0 8px var(--green); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px 16px; }
  .card h2 { font-size:13px; color:var(--dim); font-weight:500; margin-bottom:10px; text-transform:uppercase; letter-spacing:.5px; }
  .row { display:flex; justify-content:space-between; align-items:center; padding:4px 0; font-size:14px; }
  .row .k { color:var(--dim); }
  .row .v { font-family:Consolas,monospace; }
  .badge { display:inline-block; padding:2px 10px; border-radius:10px; font-size:12px; font-weight:600; }
  .badge.idle { background:#1e293b; color:var(--dim); }
  .badge.busy { background:#14352a; color:var(--green); }
  .bar { height:6px; background:#0f1419; border-radius:3px; overflow:hidden; margin-top:8px; }
  .bar > div { height:100%; background:var(--accent); transition:width .3s; border-radius:3px; }
  .queue-list { max-height:200px; overflow-y:auto; }
  .queue-item { display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid var(--border); font-size:13px; }
  .queue-item:last-child { border:0; }
  .queue-item .seq { color:var(--accent); font-family:Consolas,monospace; margin-right:8px; }
  .queue-item .fname { color:var(--txt); flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .queue-item .wait { color:var(--amber); font-family:Consolas,monospace; margin-left:8px; }
  .empty { color:var(--dim); font-size:13px; padding:8px 0; }
  .btn-mini { background:#0f1419; color:var(--red); border:1px solid var(--red); border-radius:5px;
              padding:2px 10px; font-size:12px; cursor:pointer; font-family:inherit; }
  .btn-mini:hover { background:#2a1215; }
  .upload { margin-bottom:14px; }
  .upload-form { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .upload-form input[type=file] { color:var(--dim); font-size:13px; flex:1; min-width:200px; }
  select, button { background:#0f1419; color:var(--txt); border:1px solid var(--border); border-radius:6px;
                   padding:7px 14px; font-size:13px; cursor:pointer; font-family:inherit; }
  select:focus, button:hover { border-color:var(--accent); }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); font-weight:500; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .result { margin-top:6px; }
  .result pre { background:#0f1419; border:1px solid var(--border); border-radius:6px; padding:12px;
                white-space:pre-wrap; word-break:break-word; font-size:14px; line-height:1.7;
                max-height:400px; overflow-y:auto; font-family:inherit; }
  .result .meta { color:var(--dim); font-size:12px; margin-bottom:8px; }
  .err { color:var(--red); }
  .spin { display:inline-block; width:14px; height:14px; border:2px solid var(--border);
          border-top-color:var(--accent); border-radius:50%; animation:sp .8s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
  @media(max-width:700px){ .grid{grid-template-columns:1fr} }
</style>
</head>
<body>
<h1><span class="dot" id="hd-dot"></span>Qwen3-ASR 转写控制台</h1>

<div class="card upload">
  <h2>上传转写</h2>
  <form class="upload-form" id="up-form">
    <input type="file" id="file" accept="audio/*,video/*">
    <select id="lang">
      <option value="Chinese">中文</option>
      <option value="English">English</option>
    </select>
    <button type="submit" class="primary" id="up-btn">开始转写</button>
  </form>
</div>

<div class="grid">
  <div class="card">
    <h2>当前任务</h2>
    <div id="cur-body"><div class="empty">空闲中</div></div>
  </div>
  <div class="card">
    <h2>GPU</h2>
    <div class="row"><span class="k">状态</span><span class="v" id="gpu-status">-</span></div>
    <div class="row"><span class="k">显存占用</span><span class="v" id="gpu-mem">-</span></div>
    <div class="row"><span class="k">队列长度</span><span class="v" id="qlen">0</span></div>
  </div>
</div>

<div class="card" style="margin-bottom:14px;">
  <h2>排队队列</h2>
  <div id="queue-body" class="queue-list"><div class="empty">无等待任务</div></div>
</div>

<div class="card result">
  <h2>转写状态</h2>
  <div id="res-body"><div class="empty">上传文件后,转写完成将自动下载同名 txt</div></div>
</div>

<script>
let pollTimer = null;
let myTaskKey = null;  // 上传后跟踪自己的任务(用文件名匹配)

function toSrt(segments) {
  // [[start_sec, end_sec, text], ...] → SRT 字幕文本(时间戳粒度=转写分片)
  const pad = (n, w) => String(n).padStart(w, "0");
  const ts = (sec) => {
    const ms = Math.max(0, Math.round(sec * 1000));
    return pad(Math.floor(ms / 3600000), 2) + ":" +
           pad(Math.floor((ms % 3600000) / 60000), 2) + ":" +
           pad(Math.floor((ms % 60000) / 1000), 2) + "," + pad(ms % 1000, 3);
  };
  let idx = 0;
  return segments.filter(s => s && (s[2] || "").trim()).map(s =>
    (++idx) + "\n" + ts(s[0]) + " --> " + ts(s[1]) + "\n" + s[2].trim() + "\n"
  ).join("\n");
}

async function poll() {
  try {
    const r = await fetch("/status");
    const st = await r.json();
    const dot = document.getElementById("hd-dot");
    const cur = st.current;
    document.getElementById("qlen").textContent = st.queue_len;
    document.getElementById("gpu-status").innerHTML = st.gpu.busy
      ? '<span class="badge busy">占用中</span>' : '<span class="badge idle">空闲</span>';
    document.getElementById("gpu-mem").textContent = st.gpu.mem_gb + " GB";

    const curBody = document.getElementById("cur-body");
    if (cur) {
      dot.classList.add("busy");
      const pct = cur.total_segs ? Math.round(cur.seg / cur.total_segs * 100) : 0;
      const fn = cur.audio_path.split(/[\\/]/).pop();
      curBody.innerHTML =
        '<div class="row"><span class="k">文件</span><span class="v" title="' + cur.audio_path + '">' + fn + '</span></div>' +
        '<div class="row"><span class="k">进度</span><span class="v">' + cur.seg + '/' + cur.total_segs + ' 段 (' + pct + '%)</span></div>' +
        '<div class="row"><span class="k">耗时</span><span class="v">' + cur.elapsed + 's</span></div>' +
        '<div class="row"><span class="k">语言</span><span class="v">' + cur.language + '</span></div>' +
        '<div class="bar"><div style="width:' + pct + '%"></div></div>' +
        '<div class="row" style="margin-top:8px"><span class="k"></span>' +
        '<button class="btn-mini" id="cancel-cur">终止当前任务</button></div>';
      document.getElementById("cancel-cur").addEventListener("click", () => doCancel(null));
    } else {
      dot.classList.remove("busy");
      curBody.innerHTML = '<div class="empty">空闲中</div>';
    }

    const qb = document.getElementById("queue-body");
    if (st.queue.length) {
      qb.innerHTML = st.queue.map(p => {
        const fn = p.audio_path.split(/[\\/]/).pop();
        return '<div class="queue-item"><span><span class="seq">#' + p.seq + '</span>' +
               '<span class="fname" title="' + p.audio_path + '">' + fn + '</span></span>' +
               '<span class="wait">' + p.waiting + 's ' +
               '<button class="btn-mini" data-cancel="' + p.seq + '">移除</button></span></div>';
      }).join("");
      // 排队项移除按钮:用完整 audio_path 发起取消(精确命中该项)
      qb.querySelectorAll("[data-cancel]").forEach(btn => {
        btn.addEventListener("click", () => {
          const item = st.queue.find(p => String(p.seq) === btn.dataset.cancel);
          if (item) doCancel(item.audio_path);
        });
      });
    } else {
      qb.innerHTML = '<div class="empty">无等待任务</div>';
    }
  } catch (e) { /* 忽略瞬时错误 */ }
}

async function doCancel(audioPath) {
  // audioPath 为 null = 终止当前运行任务;否则按路径子串匹配
  try {
    const body = audioPath ? { audio_path: audioPath } : {};
    const r = await fetch("/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) alert("取消失败: " + (data.detail || r.statusText));
    poll();  // 立即刷新一次状态
  } catch (err) {
    alert("取消请求失败: " + err);
  }
}

document.getElementById("up-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileEl = document.getElementById("file");
  const btn = document.getElementById("up-btn");
  const resBody = document.getElementById("res-body");
  if (!fileEl.files.length) { resBody.innerHTML = '<div class="err">请先选择文件</div>'; return; }
  const fd = new FormData();
  fd.append("file", fileEl.files[0]);
  const lang = document.getElementById("lang").value;
  btn.disabled = true;
  btn.textContent = "转写中...";
  resBody.innerHTML = '<div class="empty"><span class="spin"></span> 已上传,等待转写(可在上方查看进度)...</div>';
  try {
    const r = await fetch("/upload?language=" + encodeURIComponent(lang), { method:"POST", body: fd });
    const data = await r.json();
    if (!r.ok) {
      const d = document.createElement("div"); d.className = "err";
      d.textContent = "转写失败: " + (data.detail || r.statusText);
      resBody.replaceChildren(d); return;
    }
    const nSeg = data.segments ? data.segments.length : 0;
    // 用 Blob 触发下载:文件名取原文件名换 .txt 后缀
    const txtName = fileEl.files[0].name.replace(/\.[^.]+$/, "") + ".txt";
    const blob = new Blob([data.text || ""], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = txtName;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    const ok = document.createElement("div");
    ok.className = "meta";
    ok.textContent = "完成 · 共 " + nSeg + " 段,已下载 " + txtName;
    resBody.replaceChildren(ok);
    if (data.segments && data.segments.length) {
      const srtBtn = document.createElement("button");
      srtBtn.textContent = "下载 SRT 字幕(文本对齐录音时间)";
      srtBtn.addEventListener("click", () => {
        const blob = new Blob([toSrt(data.segments)], { type: "text/plain;charset=utf-8" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = txtName.replace(/\.txt$/, ".srt");
        document.body.appendChild(a); a.click(); a.remove();
        URL.revokeObjectURL(a.href);
      });
      resBody.appendChild(srtBtn);
    }
  } catch (err) {
    const d = document.createElement("div"); d.className = "err";
    d.textContent = "请求失败: " + err;
    resBody.replaceChildren(d);
  } finally {
    btn.disabled = false;
    btn.textContent = "开始转写";
  }
});

poll();
pollTimer = setInterval(poll, 2000);
</script>
</body>
</html>"""


# ── 上传扩展名白名单(音频+视频) ──
_UPLOAD_ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma",
                        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}


def create_app():
    """创建 FastAPI app(模型在 startup 时加载)。"""
    from fastapi import FastAPI, Request, UploadFile
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    app = FastAPI(title="Qwen3-ASR Service")

    # 防 DNS-rebinding:服务只绑 127.0.0.1,拒绝非本机 Host 的请求。
    # 注意必须是纯 ASGI 中间件,不能用 @app.middleware("http"):
    # BaseHTTPMiddleware 会包装 receive 通道,导致 request.is_disconnected()
    # 永远返回 False,转写任务的"客户端断开即取消"检测会失效。
    from urllib.parse import urlsplit

    async def _forbidden(send):
        body = b'{"detail":"forbidden host"}'
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode("latin-1"))]})
        await send({"type": "http.response.body", "body": body})

    class _HostGuardMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                host = ""
                for k, v in scope.get("headers") or []:
                    if k == b"host":
                        try:
                            host = (urlsplit("//" + v.decode("latin-1")).hostname
                                    or "").lower()
                        except ValueError:
                            host = ""
                        break
                if host not in ("localhost", "127.0.0.1", "::1"):
                    await _forbidden(send)
                    return
            await self.app(scope, receive, send)

    app.add_middleware(_HostGuardMiddleware)

    class TranscribeRequest(BaseModel):
        audio_path: str
        language: str = "Chinese"

    class CancelRequest(BaseModel):
        # 可选:传路径/文件名子串匹配终止;不传则终止当前运行任务
        audio_path: str | None = None

    @app.on_event("startup")
    async def _startup():
        global _processor, _model
        cleanup_temp_files()
        _processor, _model = load_model()
        print(f"[asr-server] 服务就绪,监听 :{PORT}", flush=True)

    @app.get("/")
    async def dashboard():
        """转写控制台(内嵌 HTML,同源访问 /status /upload)。"""
        return HTMLResponse(DASHBOARD_HTML)

    @app.get("/favicon.svg")
    async def favicon():
        """浏览器标签页图标(Q3 字样,深色主题,SVG 内嵌零文件依赖)。"""
        from fastapi.responses import Response
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
            '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
            '<stop offset="0" stop-color="#4a9eff"/>'
            '<stop offset="1" stop-color="#2563eb"/>'
            '</linearGradient></defs>'
            '<rect width="64" height="64" rx="14" fill="url(#g)"/>'
            '<text x="32" y="44" font-family="Segoe UI,Arial,sans-serif" '
            'font-size="30" font-weight="700" fill="#fff" text-anchor="middle">Q3</text>'
            '</svg>'
        )
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/health")
    async def health():
        return {"status": "ok" if _model is not None else "loading",
                "model": "qwen3-asr-1.7b"}

    @app.get("/status")
    async def status():
        """轻量状态接口:当前任务/排队/GPU 忙闲。秒回,不跑子进程。

        供人和脚本查询"在转写什么、排了几个、轮到我大概多久"。
        探活用 /health 即可(监控工具一般不解析 body)。
        """
        gpu_busy = _probe_lock_held()
        gpu_mem_gb = 0.0
        if _model is not None:
            try:
                import torch
                gpu_mem_gb = torch.cuda.memory_allocated(0) / 1073741824
            except Exception:
                pass
        with _state_lock:
            cur_raw = dict(_current_task) if _current_task else None
            pending_raw = [dict(p) for p in _pending]
        now = time.time()
        current = None
        if cur_raw:
            current = {
                "audio_path": cur_raw["audio_path"],
                "language": cur_raw["language"],
                "elapsed": round(now - cur_raw["started_at"], 1),
                "seg": cur_raw["seg"],
                "total_segs": cur_raw["total_segs"],
            }
        queue = []
        for p in pending_raw:
            queue.append({
                "seq": p["seq"],
                "audio_path": p["audio_path"],
                "language": p["language"],
                "waiting": round(now - p["arrived_at"], 1),
            })
        return {
            "status": "busy" if current else "idle",
            "current": current,
            "queue_len": len(queue),
            "queue": queue,
            "gpu": {"busy": gpu_busy, "mem_gb": round(gpu_mem_gb, 2)},
        }

    @app.post("/transcribe")
    async def transcribe(req: TranscribeRequest, request: Request):
        import asyncio
        from fastapi import HTTPException
        path = req.audio_path
        if not os.path.isfile(path):
            raise HTTPException(status_code=400, detail=f"文件不存在: {path}")
        loop = asyncio.get_running_loop()

        def client_gone() -> bool:
            """调用方断开检测(线程池 worker 中轮询,经事件循环查 socket)。"""
            fut = asyncio.run_coroutine_threadsafe(request.is_disconnected(), loop)
            try:
                return fut.result(timeout=2)
            except Exception:
                return False  # 检测本身失败不误杀,只当作未断开

        try:
            # 转写是 CPU+GPU 密集,跑在线程池避免阻塞事件循环
            result = await asyncio.to_thread(
                do_transcribe, path, req.language, client_gone)
            return result
        except TranscribeCancelled as e:
            # 被 /cancel 终止或调用方断开:499(与 nginx 的 client closed 同义,调用方需容忍)
            # 注意:此处已在事件循环线程,直接 await,不能走 client_gone()
            # (那会 schedule 回同一循环再阻塞等待,自死锁超时误报 False)
            try:
                _gone = await request.is_disconnected()
            except Exception:
                _gone = False
            print("调用方已断开,放弃转写" if _gone else str(e), flush=True)
            raise HTTPException(status_code=499, detail=str(e))
        except Exception as e:
            print(f"[asr-server] 转写失败: {e}", flush=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/cancel")
    async def cancel(req: CancelRequest):
        """终止转写任务(供人和外部项目调用)。

        传 audio_path:子串匹配,同时命中运行中+排队中的同名任务;
        排队中的立即移出队列(其转写请求在等锁轮询中发现被取消即退出)。
        不传 audio_path:终止当前运行任务。
        被终止的 /transcribe /upload 请求会收到 499。
        """
        from fastapi import HTTPException
        with _state_lock:
            cur_path = _current_task["audio_path"] if _current_task else None
        pattern = req.audio_path or cur_path
        if not pattern:
            raise HTTPException(status_code=404,
                                detail="当前无运行任务,且未指定 audio_path")
        _request_cancel(pattern)
        # 排队中命中的立即移除(界面不再显示;其线程自己也会退出)
        with _state_lock:
            matched = [p for p in _pending if pattern in p["audio_path"]]
            _pending[:] = [p for p in _pending if pattern not in p["audio_path"]]
        matched_running = cur_path is not None and pattern in cur_path
        if not matched and not matched_running:
            raise HTTPException(
                status_code=404,
                detail=f"无匹配任务(运行中/排队中均未命中): {pattern}")
        print(f"[asr-server] 收到取消: pattern={pattern!r} "
              f"命中运行中={matched_running} 移除排队={len(matched)}", flush=True)
        return {
            "cancelled": True,
            "pattern": pattern,
            "matched_running": matched_running,
            "matched_pending": [{"seq": p["seq"], "audio_path": p["audio_path"]}
                                for p in matched],
        }

    @app.post("/upload")
    async def upload(file: UploadFile, language: str = "Chinese",
                     request: Request = None):
        """上传音视频文件转写(与 /transcribe 同格式返回)。

        保存到临时文件 → 复用 do_transcribe(走同一个排队/锁机制)→ 清理。
        保留原扩展名(ffmpeg 靠扩展名识别容器格式)。
        """
        import asyncio
        from fastapi import HTTPException
        if not file.filename:
            raise HTTPException(status_code=400, detail="未提供文件")
        # 保留扩展名(白名单校验,防任意类型文件落地临时目录);无扩展名兜底 .wav
        ext = os.path.splitext(file.filename)[1].lower()
        if ext and ext not in _UPLOAD_ALLOWED_EXTS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {ext}。支持: {' '.join(sorted(_UPLOAD_ALLOWED_EXTS))}")
        if not ext:
            ext = ".wav"
        # mkstemp 而非 mktemp:消除 TOCTOU 竞态
        _fd, tmp_path = tempfile.mkstemp(suffix="_upload" + ext)
        os.close(_fd)
        try:
            # starlette 1.x 的 UploadFile 无 save(),用分块流式拷贝避免整文件入内存
            # (大视频可达 GB 级,整读会 OOM;分块峰值内存恒定 ~1MB)
            total = 0
            with open(tmp_path, "wb") as f:
                while True:
                    chunk = await file.read(1 << 20)  # 1MB
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            await file.close()
            print(f"[asr-server] 上传文件已保存: {file.filename} ({total}字节)",
                  flush=True)
        except Exception as e:
            try: os.remove(tmp_path)
            except OSError: pass
            raise HTTPException(status_code=500, detail=f"保存上传文件失败: {e}")
        try:
            # 与 /transcribe 一样跑在线程池,进同一个排队队列(可被 /status 看到)
            # 浏览器关闭/刷新页面即断开连接,转写在段间检查点自动放弃
            loop = asyncio.get_running_loop()

            def client_gone() -> bool:
                fut = asyncio.run_coroutine_threadsafe(
                    request.is_disconnected(), loop)
                try:
                    return fut.result(timeout=2)
                except Exception:
                    return False

            result = await asyncio.to_thread(
                do_transcribe, tmp_path, language, client_gone)
            return result
        except TranscribeCancelled as e:
            # 此处已在事件循环线程,直接 await(不能走 client_gone(),会自死锁)
            try:
                _gone = await request.is_disconnected()
            except Exception:
                _gone = False
            print("调用方已断开,放弃转写" if _gone else str(e), flush=True)
            raise HTTPException(status_code=499, detail=str(e))
        except Exception as e:
            print(f"[asr-server] 上传转写失败: {e}", flush=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            try: os.remove(tmp_path)
            except OSError: pass

    return app


def main():
    app = create_app()
    import uvicorn
    print(f"[asr-server] 启动 Qwen3-ASR 服务 (port={PORT})...", flush=True)
    # workers=1:GPU 天然串行,多 worker 会争显存;模型单例也要求单进程
    uvicorn.run(app, host="127.0.0.1", port=PORT, workers=1, log_level="info")


if __name__ == "__main__":
    main()
