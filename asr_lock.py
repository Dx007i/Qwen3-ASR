"""跨进程 GPU 转写互斥锁。

通过 OS 级文件锁确保同一时刻只有一个进程加载模型并占用 GPU,避免显存/
提交内存冲突导致 CUDA 崩溃(0xC0000005)。锁覆盖模型加载+推理全周期。

同一台机器上的多个转写程序(asr_server.py 与 whisper 类脚本等)共用同一把
锁文件即可互相排斥:一方持锁时,另一方要等对方推理完才能加载模型,
反之亦然,两者不会同时占 GPU。
锁文件路径默认为本文件同目录下的 .transcribe.lock,可用环境变量
ASR_LOCK_PATH 指向自定义路径(要互相排斥的多个程序指向同一个文件)。
"""
import os
import time
from pathlib import Path

# 锁文件路径:默认本文件同目录下的 .transcribe.lock(已被 .gitignore 忽略),
# 可用环境变量 ASR_LOCK_PATH 覆盖。msvcrt.locking(Windows)/fcntl.flock(Unix)
# 是 OS 级文件锁,进程被 kill 时 OS 关闭 fd 自动释放,不会死锁。
LOCK_PATH = Path(os.environ.get(
    "ASR_LOCK_PATH",
    str(Path(__file__).resolve().parent / ".transcribe.lock"),
))
# 等待锁的最长秒数。长音频转写(如 55 分钟视频)单次可跑 20 分钟以上,
# 排在其后的任务需要足够的等待上限;调用方断开时由 cancel_check(客户端
# 断开检测)提前退出,不会傻等满额。
LOCK_WAIT_TIMEOUT = 1800


class CancelledError(RuntimeError):
    """等锁期间被调用方取消(cancel_check 返回 True)时抛出。

    继承 RuntimeError 保持向后兼容;调用方可单独捕获以区分"取消"与"超时"。
    """


def acquire_transcribe_lock(cancel_check=None):
    """获取跨进程转写锁。阻塞等待,超时抛 RuntimeError。

    用 msvcrt.locking(Windows)/fcntl.flock(Linux)实现 OS 级文件锁:
    - 进程崩溃/被 kill 时,OS 关闭 fd 自动释放锁,不会死锁
    - 轮询等待(非 LK_LOCK 无限阻塞),可被 Ctrl-C 中断

    Args:
        cancel_check(callable, 可选): 等锁轮询期间每轮调用一次,返回 True
            则视为调用方已取消,立即关闭 fd 并抛 RuntimeError。
            默认 None 保持向后兼容(不检查取消)。

    Returns:
        fd(int):锁文件描述符,释放时传给 release_transcribe_lock。
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_RDWR)
    deadline = time.time() + LOCK_WAIT_TIMEOUT
    attempt = 0
    while True:
        try:
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return fd  # 获取成功
        except ImportError:
            # Linux/Mac: fcntl.flock
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError:
                pass  # 已被占用,进入等待
        except OSError:
            pass  # Windows: 已被占用,进入等待

        if time.time() > deadline:
            os.close(fd)
            raise RuntimeError(
                f"等待转写锁超时({LOCK_WAIT_TIMEOUT}秒),其他转写进程未完成"
            )
        if cancel_check is not None:
            try:
                cancelled = cancel_check()
            except Exception:
                cancelled = False
            if cancelled:
                os.close(fd)
                raise CancelledError("等待转写锁期间被取消")
        if attempt == 0:
            print("[asr-lock] 等待其他转写进程完成(GPU 互斥锁)...", flush=True)
        attempt += 1
        time.sleep(2)


def release_transcribe_lock(fd: int):
    """释放转写锁。

    Args:
        fd: acquire_transcribe_lock 返回的文件描述符。
    """
    try:
        import msvcrt
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except ImportError:
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
    except OSError:
        pass
    finally:
        os.close(fd)
