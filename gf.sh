#0.25%
python train_semisup.py --dataset IPSS \
    --num_labels 2 \
    --apply_aug classmix \
    --apply_icc \
    --snapshot_path ./saved/resnet101/gf-0.25 \
    --model_path './saved/resnet101/gf-0.25/model/model_best.pth' \
    --binary_score_map_path "./saved/resnet101/gf-0.25/pred/" \
    --gt_pred_seg_image_path "./saved/resnet101/gf-0.25/gt-pred/" \
    --mask_path "./data/datagf/gf600-0.25/test/gtfine/" \
    --log_dir './saved/resnet101/gf-0.25/test.log' \
    --Test True \
    --alpha 0.8 \
    --memco 0.5 \
    --output_dim 32 \
    --Test True 

##1%
python train_semisup.py --dataset IPSS \
    --num_labels 6 \
    --apply_aug classmix \
    --apply_icc \
    --snapshot_path ./saved/resnet101/gf-1 \
    --model_path './saved/resnet101/gf-1/model/model_best.pth' \
    --binary_score_map_path "./saved/resnet101/gf-1/pred/" \
    --gt_pred_seg_image_path "./saved/resnet101/gf-1/gt-pred/" \
    --mask_path "./data/datagf/gf600-1/test/gtfine/" \
    --log_dir './saved/resnet101/gf-1/test.log' \
    --Test True \
    --alpha 0.8 \
    --memco 0.5 \
    --output_dim 32 \
    --Test True 

# #2%
python train_semisup.py --dataset IPSS \
    --num_labels  12 \
    --apply_aug classmix \
    --apply_icc \
    --snapshot_path ./saved/resnet101/gf-2 \
    --model_path './saved/resnet101/gf-2/model/model_best.pth' \
    --binary_score_map_path "./saved/resnet101/gf-2/pred/" \
    --gt_pred_seg_image_path "./saved/resnet101/gf-2/gt-pred/" \
    --mask_path "./data/datagf/gf600-2/test/gtfine/" \
    --log_dir './saved/resnet101/gf-2/test.log' \
    --Test True \
    --alpha 0.7 \
    --memco 0.5 \
    --output_dim 64 \
    --Test True 
