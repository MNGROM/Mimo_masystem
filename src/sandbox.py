import os
import subprocess
import tempfile
import hashlib
import shutil
import errno
from pathlib import Path


def _hash_compile_key(code: str, gpp: str, std: str) -> str:
    h = hashlib.sha1()
    h.update(gpp.encode("utf-8"))
    h.update(b"\0")
    h.update(std.encode("utf-8"))
    h.update(b"\0")
    h.update(code.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def compile_cpp(code: str, gpp: str, std: str, timeout_sec: int):
    """
    Compile C++ code into a temporary directory and return (bin_path, ce_msg).
    """
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "main.cpp")
        out = os.path.join(td, "main.out")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [gpp, f"-std={std}", "-O2", "-pipe", src, "-o", out]
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return None, "CE: compile timeout"
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="ignore")[:4000]
            return None, f"CE: {err}\nCMD: {' '.join(cmd)}"

        try:
            os.chmod(out, os.stat(out).st_mode | 0o111)
        except Exception:
            pass
        return out, None


def compile_cpp_cached(code: str, gpp: str, std: str, timeout_sec: int, cache_dir: str):
    """
    Compile C++ with a persistent on-disk cache keyed by (gpp,std,code).
    Returns (bin_path, ce_msg).

    IMPORTANT:
      Some environments mount /tmp on a different filesystem than your project dir.
      In that case, os.replace(/tmp/... -> ./.compile_cache/...) raises EXDEV.
      We fallback to copy+atomic replace inside the cache dir.
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _hash_compile_key(code, gpp=gpp, std=std)
    bin_path = cache_root / f"{key}.out"
    err_path = cache_root / f"{key}.ce.txt"

    # If binary already exists and is executable, reuse it.
    if bin_path.exists():
        try:
            bin_path.chmod(bin_path.stat().st_mode | 0o111)
        except Exception:
            pass
        return str(bin_path), None

    # If a previous compile error exists, reuse that (saves time on repeated bad code)
    if err_path.exists():
        try:
            return None, "CE_CACHED: " + err_path.read_text(encoding="utf-8")[:4000]
        except Exception:
            pass

    # Compile into a temp dir then move into cache
    with tempfile.TemporaryDirectory() as td:
        src_tmp = os.path.join(td, "main.cpp")
        out_tmp = os.path.join(td, "main.out")
        with open(src_tmp, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [gpp, f"-std={std}", "-O2", "-pipe", src_tmp, "-o", out_tmp]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            msg = "compile timeout"
            try:
                err_path.write_text(msg, encoding="utf-8")
            except Exception:
                pass
            return None, "CE: " + msg
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="ignore")[:4000]
            try:
                err_path.write_text(err, encoding="utf-8")
            except Exception:
                pass
            return None, f"CE: {err}\nCMD: {' '.join(cmd)}"

        # Move compiled binary into cache.
        # NOTE: /tmp may be on a different filesystem than cache_dir (common on cloud setups),
        # so os.replace() can fail with EXDEV. We fallback to copy+atomic replace within cache.
        try:
            os.replace(out_tmp, str(bin_path))
        except OSError as e:
            if e.errno == errno.EXDEV:
                tmp_in_cache = str(bin_path) + f".tmp.{os.getpid()}"
                shutil.copy2(out_tmp, tmp_in_cache)
                os.replace(tmp_in_cache, str(bin_path))  # now same dir -> atomic
                try:
                    os.remove(out_tmp)
                except FileNotFoundError:
                    pass
            else:
                raise

        try:
            bin_path.chmod(bin_path.stat().st_mode | 0o111)
        except Exception:
            pass
        return str(bin_path), None


def run_bin(bin_path: str, inp: str, timeout_sec: int, max_output_bytes: int):
    """
    IMPORTANT: builders.py expects 3 return values:
      (returncode, stdout_str, stderr_str)

    - On TLE: returncode = -1, stdout="", stderr="TLE"
    """
    try:
        p = subprocess.run(
            [bin_path],
            input=inp.encode("utf-8", errors="ignore"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        return -1, "", "TLE"

    out = p.stdout[:max_output_bytes].decode("utf-8", errors="ignore")
    err = p.stderr.decode("utf-8", errors="ignore")[:4000]
    return p.returncode, out, err