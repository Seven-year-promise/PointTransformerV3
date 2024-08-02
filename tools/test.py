"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    args = default_argument_parser().parse_args()
    args.config_file = "/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/config.py"
    args.save_path = "exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3"
    args.num_gpus = 1
    args.num_machines = 1
    cfg = default_config_parser(args.config_file, args.save_path, args.options)

    print(cfg)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
