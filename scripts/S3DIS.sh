# S3DIS experiment scripts
# We provide 0.1% and 0.01% weakly setting experiment scripts

# 0.1% baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-3_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0. \
    AUGMENTATION.use_color_jitter False

# 0.1% consistency baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-3_consis_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 1. \
    AUGMENTATION.use_color_jitter False

# 0.1% CPCM
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_CPCM \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_grid_size 4 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_loss_weight 5. \
    TRAINER.two_stream_loss_mask_weight 5. \
    AUGMENTATION.use_color_jitter False

# 0.01% baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 0. \
    AUGMENTATION.use_color_jitter False

# 0.01% consistency baseline
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_consis_baseline \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_loss_weight 5. \
    AUGMENTATION.use_color_jitter False


# 0.01% CPCM
CUDA_VISIBLE_DEVICES=0 python launch.py ddp_train.py --config config/default.yaml \
    GENERAL.exp_name wfs_percentage1e-4_CPCM \
    TRAINER.name TwoStreamTrainer \
    MODEL.out_channels 13 \
    DATA.name StanfordDataLoader \
    DATA.dataset StanfordArea5Dataset \
    DATA.voxel_size 0.05 \
    DATA.batch_size 2 \
    DATA.train_limit_numpoints 1000000 \
    OPTIMIZER.lr 0.01 \
    OPTIMIZER.weight_decay 0.001 \
    SCHEDULER.name PolyLR \
    TRAINER.epochs 180 \
    EVALUATOR.iou_num_class 13 \
    DATA.stanford3d_path ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed \
    DATA.stanford3d_sampled_inds ${YOUR_PATH_TO}/stanford_fully_supervised_preprocessed/points/percentage0.0001evenc \
    DATA.sparse_label False \
    DATA.two_stream True \
    MODEL.two_stream_model_apply True \
    TRAINER.two_stream_feats_key semantic_scores \
    TRAINER.two_stream_loss_mode js_divergence_v2 \
    TRAINER.two_stream_seg_both True \
    TRAINER.two_stream_mask_extra_stream True \
    TRAINER.two_stream_mask_grid_size 8 \
    TRAINER.two_stream_mask_ratio 0.75 \
    TRAINER.two_stream_loss_weight 2. \
    TRAINER.two_stream_loss_mask_weight 10. \
    AUGMENTATION.use_color_jitter False