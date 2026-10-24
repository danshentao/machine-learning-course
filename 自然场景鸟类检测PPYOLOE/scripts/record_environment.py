"""保存本机软件、GPU、依赖和官方仓库提交信息，避免手工抄录错误。"""
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "PaddleDetection"


def cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def main():
    import paddle
    import cv2
    import numpy
    import PIL
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "paddle": paddle.__version__,
        "paddle_cuda_compiled": paddle.is_compiled_with_cuda(),
        "paddle_cuda_device_count": paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0,
        "numpy": numpy.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
        "paddledetection_commit": cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "nvidia_smi_hardware": cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "nvcc": cmd(["nvcc", "--version"]),
        "config": str(ROOT / "configs" / "ppyoloe_s_bird_640.yml"),
        "pretrained_checkpoint_source": "https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_300e_coco.pdparams",
    }
    (ROOT / "reports" / "environment_record.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze = cmd([sys.executable, "-m", "pip", "freeze"])
    (ROOT / "reports" / "requirements_frozen.txt").write_text(freeze + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
