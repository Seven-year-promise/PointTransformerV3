"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    args.config_file = "configs/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top.py"
    args.save_path = "exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3"
    cfg = default_config_parser(args.config_file, args.save_path, args.options)

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
