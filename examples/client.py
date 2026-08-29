#!/usr/bin/env python3
"""Qwen3-ASR 服务调用示例(需 pip install requests)。

用法:
  python client.py E:/media/lecture.mp4                        # 同机路径转写
  python client.py E:/media/lecture.mp4 --upload               # 改为上传文件
  python client.py E:/media/lecture.mp4 --server http://192.168.1.10:8003
  python client.py E:/media/lecture.mp4 --language English --out result

输出转写文本到终端;--out 指定前缀时另存 <out>.txt 与 <out>.srt 字幕。
"""
import argparse
import sys

import requests


def to_srt(segments):
    """[[start_sec, end_sec, text], ...] → SRT 字幕文本(时间粒度=28s 切片)。"""
    def ts(sec):
        ms = max(0, round(sec * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = [
        f"{i}\n{ts(a)} --> {ts(b)}\n{t.strip()}\n"
        for i, (a, b, t) in enumerate(segments, 1) if (t or "").strip()
    ]
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-ASR 服务调用示例")
    ap.add_argument("media", help="音视频文件路径")
    ap.add_argument("--server", default="http://127.0.0.1:8003", help="服务地址")
    ap.add_argument("--language", default="Chinese", help="语言(默认 Chinese)")
    ap.add_argument("--upload", action="store_true",
                    help="上传文件转写(默认走 /transcribe 传本地路径)")
    ap.add_argument("--out", help="结果保存前缀,如 --out result 保存 result.txt/result.srt")
    args = ap.parse_args()

    # 连接超时 10s、读取不限时——转写是同步长请求,读取设短了等于主动取消任务
    if args.upload:
        with open(args.media, "rb") as f:
            r = requests.post(
                f"{args.server}/upload", params={"language": args.language},
                files={"file": f}, timeout=(10, None))
    else:
        r = requests.post(
            f"{args.server}/transcribe",
            json={"audio_path": args.media, "language": args.language},
            timeout=(10, None))

    if r.status_code == 499:
        sys.exit("任务被取消(499)")
    if r.status_code != 200:
        try:
            sys.exit(f"失败[{r.status_code}]: {r.json().get('detail')}")
        except ValueError:
            r.raise_for_status()

    data = r.json()
    print(data["text"])
    if args.out:
        with open(args.out + ".txt", "w", encoding="utf-8") as f:
            f.write(data["text"])
        with open(args.out + ".srt", "w", encoding="utf-8") as f:
            f.write(to_srt(data["segments"]))
        print(f"\n已保存 {args.out}.txt 与 {args.out}.srt", file=sys.stderr)


if __name__ == "__main__":
    main()
