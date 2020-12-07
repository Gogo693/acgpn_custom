nohup python train.py --dataroot ../../ACGPN_landmarks/ACGPN_train/data/ --gpu_ids 3 --name prova &
nohup python test.py --dataroot data/Data_preprocessing --gpu_ids 0 &
nohup python test.py --dataroot ../../ACGPN_landmarks_mesh/ACGPN_inference/data/Data_preprocessing/ --gpu_ids 3 --name prova &
nohup python test.py --dataroot ../../ACGPN_landmarks/ACGPN_inference/data/Data_preprocessing/ --gpu_ids 1 --name dense --dense &
