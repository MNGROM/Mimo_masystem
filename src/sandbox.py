import os
import subprocess
import tempfile

def compile_cpp(code: str, gpp: str, std: str, timeout_sec: int):
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, "main.cpp")
        bin_path = os.path.join(td, "main.out")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [gpp, f"-std={std}", "-O2", "-pipe", src_path, "-o", bin_path]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return None, "CE: compile timeout"
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode("utf-8", errors="ignore")[:4000]
            return None, f"CE: {err}"

        final_bin = tempfile.NamedTemporaryFile(delete=False)
        final_bin.close()
        os.unlink(final_bin.name)
        os.rename(bin_path, final_bin.name)
        return final_bin.name, None

def run_bin(bin_path: str, inp: str, timeout_sec: int, max_output_bytes: int):
    try:
        p = subprocess.run(
            [bin_path],
            input=inp.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
        )
        out = p.stdout[:max_output_bytes].decode("utf-8", errors="ignore")
        err = p.stderr[:max_output_bytes].decode("utf-8", errors="ignore")
        return p.returncode, out, err
    except subprocess.TimeoutExpired:
        return -1, "", "TLE"