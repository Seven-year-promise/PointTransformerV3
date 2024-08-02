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
import open3d as o3d
import numpy as np
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS


class Seg3D(object):
    def __init__(self, config_file, save_path) -> None:
        args = default_argument_parser().parse_args()
        args.config_file = config_file
        args.save_path = save_path
        args.num_gpus = 1
        args.num_machines = 1
        cfg = default_config_parser(args.config_file, args.save_path, args.options)

        cfg = default_setup(cfg)
        self.tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))

    def predict(self, pcd_array:np.array) -> np.array:
        pred = self.tester.predict(new_pcd_array)

        return pred

        



if __name__ == "__main__":
    segmentor = Seg3D(config_file = "/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3/config.py",
                      save_path = "exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3")

    NUM = "000187"

    pcd_data = o3d.io.read_point_cloud("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pcd/avia_187.pcd")
    pcd_array = np.array(pcd_data.points)

    #pcd_array = np.fromfile("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/bin/avia_100.bin", dtype=np.float32)#.reshape(-1, 3)
    #pcd_array = np.fromfile("/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/velodyne/" + NUM + ".bin", dtype=np.float32).reshape(-1, 3)

    #new_pcd_array = pcd_array[np.any(pcd_array != 0, axis=1)]

    # pcd_array = np.fromfile("/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/velodyne/000099.bin", dtype=np.float32).reshape(-1, 3)
    new_pcd_array = pcd_array[np.any(pcd_array != 0, axis=1)]
    pred = segmentor.predict(new_pcd_array)
    print(max(pred))


    np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3/txt/" + NUM + "inf.txt", new_pcd_array, delimiter=',')

    pcd_array_cp = new_pcd_array.copy()
    pcd_array_cp[np.where(pred!=1)] = [0, 0, 0]
    np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3/txt/" + NUM + "inf_pred1.txt", pcd_array_cp, delimiter=',')

    pcd_array_cp2 = new_pcd_array.copy()
    pcd_array_cp2[np.where(pred!=2)] = [0, 0, 0]
    np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3/txt/" + NUM + "inf_pred2.txt", pcd_array_cp2, delimiter=',')
    

