"""调用原版 PaddleDetection tools/train.py；只把 --enable_ce 默认种子替换为配置种子123。"""
from __future__ import annotations
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
CFG=json.loads((HERE/'paths.json').read_text(encoding='utf-8'))
REPO=Path(CFG['paddledetection'])
sys.path.insert(0,str(REPO))

def main():
    import numpy as np
    import paddle
    from ppdet.engine.env import set_random_seed
    seed=int(CFG['seed'])
    random.seed(seed);np.random.seed(seed);paddle.seed(seed)
    official=REPO/'tools/train.py'
    spec=importlib.util.spec_from_file_location('paddledetection_official_train',official)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # 原版入口把 --enable_ce=True 固定成seed=0；在内存中适配这一处，不改第三方源码。
    module.set_random_seed=lambda _ignored:set_random_seed(seed)
    sys.argv=[str(official),*sys.argv[1:]]
    module.main()

if __name__=='__main__':main()
