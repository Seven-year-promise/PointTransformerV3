SemanticKITTI
https://apc01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fs3.eu-central-1.amazonaws.com%2Favg-kitti%2Fdata_odometry_velodyne.zip&data=05%7C02%7Cyankewang%40ust.hk%7C2e68d69fbf6c456679fe08dca7127147%7Cc917f3e2932249269bb3daca730413ca%7C1%7C0%7C638568944861617944%7CUnknown%7CTWFpbGZsb3d8eyJWIjoiMC4wLjAwMDAiLCJQIjoiV2luMzIiLCJBTiI6Ik1haWwiLCJXVCI6Mn0%3D%7C0%7C%7C%7C&sdata=0LsHXVckP4AMwq%2BPZAf0i2tJH9kFZpqpnBI48phx08s%3D&reserved=0


NuScenes
https://www.nuscenes.org/nuscenes#download

ln -s home/tower_crane_data/3d_dataset/SemanticKITTI/SemanticKITTI/dataset /home/HKCRC_perception/PC_gen/PointTransformerV3/data/semantic_kitti

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=./
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
python tools/train.py --config-file configs/semantic_kitti/semseg-pt-v2m2-1-benchmark-submit.py --options save_path=exp/semantic_kitti/semseg-pt-v2m2-1-benchmark-submit


python tools/train.py --config-file configs/semantic_kitti/semseg-pt-v2m2-0-base.py --options save_path=exp/semantic_kitti/semseg-pt-v2m2-0-base


export PYTHONPATH=./
sh scripts/test.sh -p python -d semantic_kitti -n semseg-pt-v2m2-1-benchmark-submit -w model_last -g 1

python tools/test.py --config-file configs/semantic_kitti/semseg-pt-v2m2-1-benchmark-submit.py --options save_path=/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-1-benchmark-submit/model/model_last.pth


### Training for construction top

python tools/train.py --config-file configs/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top.py --options save_path=exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top

python tools/test.py --config-file configs/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top.py --options save_path=/home/HKCRC_perception/PC_gen/PointTransformerV3/exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top/model/model_last.pth

sh scripts/test.sh -p python -d semantic_kitti -n semseg-pt-v2m2-0-base_crcust_top -w model_last -g 1


PYTHON -u tools/test.py  --config-file "$CONFIG_DIR" \
  --num-gpus "$GPU" \
  --options save_path="$EXP_DIR" weight="${MODEL_DIR}"/"${WEIGHT}".pth



## 3 classes

python tools/train.py --config-file configs/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top.py --options save_path=exp/semantic_kitti/semseg-pt-v2m2-0-base_crcust_top_cls3

sh scripts/test.sh -p python -d semantic_kitti -n semseg-pt-v2m2-0-base_crcust_top_cls3 -w model_last -g 1