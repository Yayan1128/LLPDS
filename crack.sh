
#%0.25
nohup python train_semisup.py --dataset crack \
        --num_labels 5 \
        --apply_aug classmix \
        --apply_icc   \
        --memco 0.3 \
        --snapshot_path   /home/wyy/PycharmProjects/ss/pavement/LLPD/saved/resnet101/crack0.25  \
        --model_path '/home/wyy/PycharmProjects/ss/pavement/LLPD/saved/resnet101/crack0.25/model/model_best.pth'  \
       --binary_score_map_path  "/home/wyy/PycharmProjects/ss/pavement/LLPD/saved/resnet101/crack0.25/pred/"  \
       --gt_pred_seg_image_path "/home/wyy/PycharmProjects/ss/pavement/LLPD/saved/resnet101/crack0.25/gt-pred/" \
       --mask_path  "/home/wyy/PycharmProjects/ss/CRACK500/crack0.25/test/gtfine/"  \
       --log_dir '/home/wyy/PycharmProjects/ss/pavement/LLPD/saved/resnet101/crack0.25/test.log'\
       --Test True  &
