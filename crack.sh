
#%0.25
python train_semisup.py --dataset crack \
        --num_labels 5 \
        --apply_aug classmix \
        --apply_icc   \
        --memco 0.5 \
        --snapshot_path   ./saved/resnet101/crack-0.25  \
        --model_path './saved/resnet101/crack-0.25/model/model_best.pth'  \
       --binary_score_map_path  "./saved/resnet101/crack-0.25/pred/"  \
       --gt_pred_seg_image_path "./saved/resnet101/crack-0.25/gt-pred/" \
       --mask_path  "./data/crack0.25/test/gtfine/"  \
       --log_dir './saved/resnet101/crack-0.25/test.log'\
       --Test True  


# #1%
python train_semisup.py --dataset crack \
        --num_labels 19 \
        --apply_aug classmix \
        --apply_icc   \
        --memco 0.3 \
        --snapshot_path   ./saved/resnet101/crack-1  \
        --model_path './saved/resnet101/crack-1/model/model_best.pth'  \
       --binary_score_map_path  "./saved/resnet101/crack-1/pred/"  \
       --gt_pred_seg_image_path "./saved/resnet101/crack-1/gt-pred/" \
       --mask_path  "./data/crack-1/test/gtfine/"  \
       --log_dir './saved/resnet101/crack-1/test.log'\
       --Test True  

# #2%      
python train_semisup.py --dataset crack \
        --num_labels 38 \
        --apply_aug classmix \
        --apply_icc   \
        --memco 0.1 \
        --alpha 0.8 \
        --snapshot_path   ./saved/resnet101/crack-2  \
        --model_path './saved/resnet101/crack-2/model/model_best.pth'  \
       --binary_score_map_path  "./saved/resnet101/crack-2/pred/"  \
       --gt_pred_seg_image_path "./saved/resnet101/crack-2/gt-pred/" \
       --mask_path  "./data/crack-2/test/gtfine/"  \
       --log_dir './saved/resnet101/crack-2/test.log'\
       --Test True  
