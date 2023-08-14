# Semantic Kitii FoV experiment scripts
# We provide 0.1% and 0.01% weakly setting experiment scripts

# 0.1%, baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0

# 0.1%, consistency baseline, consis weight 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_consis_weight5 \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 5.0

# 0.1%, CPCM, consis weight 2, mask mode grid, mask grid size 8, mask ratio 0.75, mask weight 10.0
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-3_percentage_consis_weight2_maskGrid075GridSize8_weight10 \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 2.0 \
    TRAINER.two_stream_mask_grid_size 8 \
    TRAINER.two_stream_loss_mask_mode js_divergence_v2 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_mask_mode grid \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_feats_key semantic_scores \
    TRAINER.two_stream_mask_corr_loss True \
    TRAINER.two_stream_mask_self_loss True \
    TRAINER.two_stream_loss_mask_weight 10. \
    TRAINER.two_stream_mask_loss_threshold -1.0

# 0.01%, baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0.0

# 0.01%, consistency baseline, consis weight 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_consis_weight5 \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 5.0

# 0.01%, CPCM, consis weight 2, mask mode grid, mask grid size 8, mask ratio 0.75, mask weight 10.0
CUDA_VISIBLE_DEVICES=0,1,2,3 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name 1e-4_percentage_consis_weight2_maskGrid075GridSize8_weight10 \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 19 \
    DATA.name SemanticKittiDataLoader \
    DATA.dataset SemanticKittiVoxelization2cmDataset \
    OPTIMIZER.lr 0.01 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 50 \
    DATA.train_limit_numpoints 1000000 \
    DATA.batch_size 8 \
    TRAINER.empty_cache_every 1 \
    EVALUATOR.iou_num_class 19 \
    AUGMENTATION.normalize_color True \
    DATA.semantic_kitti_path ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed \
    DATA.semantic_kitti_sampled_inds ${YOUR_PATH_TO}/semantic_kitti_fov_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 2.0 \
    TRAINER.two_stream_mask_grid_size 8 \
    TRAINER.two_stream_loss_mask_mode js_divergence_v2 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_mask_mode grid \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_feats_key semantic_scores \
    TRAINER.two_stream_mask_corr_loss True \
    TRAINER.two_stream_mask_self_loss True \
    TRAINER.two_stream_loss_mask_weight 10. \
    TRAINER.two_stream_mask_loss_threshold -1.0
