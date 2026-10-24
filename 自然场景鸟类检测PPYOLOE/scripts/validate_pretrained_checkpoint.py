"""在CPU上检查官方PP-YOLOE-s权重能否初始化8类模型。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "PaddleDetection"


def main() -> None:
    # Paddle GPU wheel 即使选择CPU设备，也会在建卷积层时查询cuDNN版本。
    # 将虚拟环境自带的DLL目录加入当前进程，不改系统PATH。
    nvidia = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    for dll_dir in nvidia.glob("*/bin"):
        os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(dll_dir))
    sys.path.insert(0, str(REPO))

    import paddle
    from ppdet.core.workspace import create, load_config
    from ppdet.utils.checkpoint import load_pretrain_weight

    paddle.set_device("cpu")
    cfg = load_config(str(ROOT / "configs" / "ppyoloe_s_bird_640.yml"))
    model = create(cfg.architecture)
    load_pretrain_weight(model, cfg.pretrain_weights)
    print(f"checkpoint_load=passed architecture={cfg.architecture} classes={cfg.num_classes} device=cpu")


if __name__ == "__main__":
    main()
