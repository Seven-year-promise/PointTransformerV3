from pathlib import Path
import shutil
import numpy as np
import scipy
import struct

def read_bin(data_path):
    points = []
    # Open the binary file
    with open(data_path, "rb") as file:
        while True:
            # Read 4 floats (16 bytes) at a time
            bytes = file.read(16)
            if not bytes:
                break
            # Unpack the bytes to floats
            x, y, z, intensity = struct.unpack('ffff', bytes)
            points.append([x, y, z, intensity])

    points = np.array(points)

    return points

label_path = Path("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/3d_seg_anno/VoxelLabelData_1")
bin_path = Path("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/bin")
pcd_path = Path("/home/tower_crane_data/crcust/crcust_top/mvs_avia/result_5/pcd")

save_pcd_path = Path("/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/velodyne")
save_label_path = Path("/home/tower_crane_data/crcust/dataset/crcust_top_3d_seg/v1/dataset/sequences/00/labels")

# # Create the folders if they don't exist
save_pcd_path.mkdir(parents=True, exist_ok=True)
save_label_path.mkdir(parents=True, exist_ok=True)

# # Get a list of files in the folder
label_files = sorted(label_path.rglob('*.mat'), key=lambda a: int(a.stem.split("_")[-1]))
bin_files = sorted(bin_path.rglob('*.bin'), key=lambda a: int(a.stem.split("_")[-1]))
pcd_files = sorted(pcd_path.rglob('*.pcd'), key=lambda a: int(a.stem.split("_")[-1]))
# # Iterate through each file
# for file in pcd_files:
#     idx = str(file.name.split(".")[0].split("_")[-1])
#     new_name = str(idx).zfill(6) + file.suffix
#     shutil.copy(file, save_pcd_path / new_name)

for b_f, l_f, p_f in zip(bin_files, label_files, pcd_files):
    mat_data = scipy.io.loadmat(l_f)

    # Access the data from the loaded .mat file
    # For example, if the variable in the .mat file is named 'data'
    data = mat_data['L']

    # Convert the data to a NumPy array
    data_np = np.array(data)

    bin_data = data_np[:, :-1].astype(np.float32)
    label_data = data_np[:, -1].astype(np.int32)

    max_label = max(label_data)

    if max(label_data) > 0:

        idx = str(b_f.name.split(".")[0].split("_")[-1])

        new_b_f = str(idx).zfill(6) + b_f.suffix
        new_l_f = str(idx).zfill(6) + ".label"

        # shutil.copy(b_f, save_pcd_path / new_b_f)

        new_bin_data = bin_data[np.any(bin_data != 0, axis=1)]
        new_label_data = label_data[np.any(bin_data != 0, axis=1)]
        new_label_data.tofile(str(save_label_path / new_l_f))
        new_bin_data.tofile(str(save_pcd_path / new_b_f))

        # np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/test/ori.txt", new_bin_data, delimiter=',')
        # new_bin_data[np.where(new_label_data!=1)] = [0, 0, 0]
        # np.savetxt("/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/test/pred.txt", new_bin_data, delimiter=',')

        # break

    

