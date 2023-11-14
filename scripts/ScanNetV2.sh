# ScanNet V2 experiment scripts
# We provide 0.1% and 0.01% weakly setting experiment scripts

# 0.1% baseline
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0 \
    TRAINER.empty_cache_every 1

# 0.1% consistency baseline, consis weight 1
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_consis_weight1 \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 1.0 \
    TRAINER.empty_cache_every 1

# 0.1% CPCM, consis weight 0, mask mode grid, mask grid size 4, mask ratio 0.75, mask weight 5
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_consis_weight0_maskGrid075GridSize4_weight5 \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter True \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0 \
    TRAINER.two_stream_loss_threshold -1. \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_mask_mode grid \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_feats_key semantic_scores \
    TRAINER.two_stream_loss_mask_mode js_divergence_v2 \
    TRAINER.two_stream_mask_corr_loss True \
    TRAINER.two_stream_mask_self_loss True \
    TRAINER.two_stream_loss_mask_weight 5. \
    TRAINER.two_stream_mask_loss_threshold -1. \
    TRAINER.two_stream_mask_grid_size 4 \
    TRAINER.empty_cache_every 1

# 0.01% baseline
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_baseline \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0 \
    TRAINER.empty_cache_every 1

# 0.01% consistency baseline, consis weight 5.0
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_consis_weight5 \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 5.0 \
    TRAINER.empty_cache_every 1

# 0.01% CPCM, consis weight 0, mask mode grid, mask grid size 8, mask ratio 0.75, mask weight 10
CUDA_VISIBLE_DEVICES=0,1 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_consis_weight0_maskGrid075GridSize8_weight10 \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
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
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    AUGMENTATION.use_color_jitter False \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0 \
    TRAINER.two_stream_loss_threshold -1. \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_mask_mode grid \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_feats_key semantic_scores \
    TRAINER.two_stream_loss_mask_mode js_divergence_v2 \
    TRAINER.two_stream_mask_corr_loss True \
    TRAINER.two_stream_mask_self_loss True \
    TRAINER.two_stream_loss_mask_weight 10. \
    TRAINER.two_stream_mask_loss_threshold -1. \
    TRAINER.two_stream_mask_grid_size 8 \
    TRAINER.empty_cache_every 1

# for evaluating scannet testset
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name evaluate_scannet_testset \
    TRAINER.name TwoStreamTrainer \
    DATA.dataset ScannetVoxelization2cmDataset \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    DATA.batch_size 2 \
    DATA.scannet_path ${YOUR_PATH_TO}/scannet_fully_supervised_preprocessed \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_seg_both True \
    TRAINER.run_scannet_test True \
    TRAINER.scannet_testset_output_result_path ${YOUR_PATH_TO_SCANNET_TESTSET_OUTPUT_RESULT} \
    MODEL.resume True \
    MODEL.resume_path ${YOUR_PATH_TO_RESUME_MODEL}

