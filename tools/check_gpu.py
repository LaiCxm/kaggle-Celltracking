"""check_gpu.py — 一键检测本地是否具备独立 NVIDIA 显卡。

用法:
  python tools/check_gpu.py

输出示例:
  [OK] 本地有算力: NVIDIA GeForce RTX 4090 (24GB), torch.cuda.is_available=True -> 优先本地训练
  [INFO] 本地无独显 -> 走 Kaggle GPU 推送流程
"""
import shutil
import subprocess
import sys

def check_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi 未找到"
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return False, f"nvidia-smi 执行失败: {r.stderr.strip()[:200]}"
        lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return False, "nvidia-smi 无输出"
        # 解析显存，取最大值
        max_mem = 0
        for line in lines:
            # 格式如 "NVIDIA GeForce RTX 4090, 24564 MiB"
            if "MiB" in line:
                try:
                    mem = int(line.split(",")[-1].strip().split()[0])
                    max_mem = max(max_mem, mem)
                except Exception:
                    pass
        return True, f"nvidia-smi 检测到: {lines[0]} (最大显存 {max_mem} MiB)"
    except Exception as e:
        return False, f"nvidia-smi 异常: {e}"

def check_torch():
    try:
        import torch
        avail = torch.cuda.is_available()
        if not avail:
            return False, f"torch {torch.__version__} 但 torch.cuda.is_available()=False"
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            return True, f"torch {torch.__version__} cuda可用: {name} ({total:.0f} MiB)"
        except Exception as e:
            return True, f"torch {torch.__version__} cuda可用 (获取设备名失败: {e})"
    except ImportError:
        return False, "torch 未安装"
    except Exception as e:
        return False, f"torch 检测异常: {e}"

def main():
    print("=== 本地 GPU 自检 ===")
    ok_smi, msg_smi = check_nvidia_smi()
    print(f"[{'OK' if ok_smi else 'INFO'}] {msg_smi}")
    ok_torch, msg_torch = check_torch()
    print(f"[{'OK' if ok_torch else 'INFO'}] {msg_torch}")

    # 综合判定：需 nvidia-smi 成功且 torch 可用且显存 >=4GB
    has_gpu = ok_smi and ok_torch
    # 额外检查显存阈值（从 nvidia-smi 解析）
    if has_gpu:
        try:
            import re
            m = re.search(r"(\d+)\s*MiB", msg_smi)
            if m and int(m.group(1)) < 4096:
                print(f"[WARN] 显存 <4GB ({m.group(1)} MiB)，建议仍走 Kaggle GPU")
                has_gpu = False
        except Exception:
            pass

    print()
    if has_gpu:
        print("[OK] 本地有算力 -> 优先本地训练/推理，仅评分/提交走 Kaggle API")
        print("     示例: python EXP/EXP001/train.py --config EXP/EXP001/config/child-exp000.yaml --folds 0,1,2,3,4")
    else:
        print("[INFO] 本地无独显或不满足条件 -> 走 Kaggle GPU 推送流程")
        print("     示例: python tools/push_training.py --exp EXP001 --child child-exp000 --competition biohub-cell-tracking-during-development")
    sys.exit(0 if has_gpu else 1)

if __name__ == "__main__":
    main()
