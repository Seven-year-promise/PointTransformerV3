import os
import open3d as o3d
import numpy as np

def convert_pcd_to_bin(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.pcd'):
            pcd_path = os.path.join(input_folder, filename)
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points, dtype=np.float32)
            bin_filename = os.path.splitext(filename)[0] + '.bin'
            bin_path = os.path.join(output_folder, bin_filename)
            points.tofile(bin_path)
            print(f"Converted {pcd_path} to {bin_path}")

input_folder = '/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pcd'
output_folder = '/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/bin'
convert_pcd_to_bin(input_folder, output_folder)

