# ScanNet V2 experiment scripts
# We provide 0.1% and 0.01% weakly setting experiment scripts

# 0.1% baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-3_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0

# 0.1% consistency baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-3_consis_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 1.0

# 0.1% CPCM
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-3_CPCM \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_grid_size 4 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_loss_weight 1. \
    TRAINER.two_stream_loss_mask_weight 5.0


# 0.01% baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0

# 0.01% consistency baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 5.0

# 0.01% CPCM
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.scannet_sampled_inds ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_grid_size 8 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_loss_weight 10. \
    TRAINER.two_stream_loss_mask_weight 5.0
