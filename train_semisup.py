import torch
import torchvision.models as models
import torch.optim as optim
import argparse

from network.deeplabv3.deeplabv3 import *
from network.deeplabv2 import *
from build_data import *
from module_list import *

from Utils.pyt_utils import load_model
from uutils import *
import torch.backends.cudnn as cudnn
from utils.init_func import init_weight
# import wandb
from LPURf import LPUR
import time
# wandb.init(project="Teacher-student-abmemco", entity="wyy-team",name="LLPD-0.25")

def save_seg_results(binary_scores, img_name, binary_score_map_path, gt_pred_seg_image_path, mask_path):

    img_name1 = '{}{}'.format(img_name, '.bmp')

    pre = (binary_scores * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(binary_score_map_path, "{}".format(img_name1)), pre)

    # pred vs gt image
    visulization(img_file=img_name, mask_path=mask_path,
                 score_map_path=binary_score_map_path, saving_path=gt_pred_seg_image_path)
def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

def defreeze(model: nn.Module):
        """Freeze the model."""
        model.train()
        for param in model.parameters():
            param.requires_grad = True


parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Perfect Labels')
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_labels', default=15, type=int, help='number of labelled training data, set 0 to use all training data')
parser.add_argument('--lr', default=2.5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--dataset', default='cityscapes', type=str, help='pascal, cityscapes, sun')
parser.add_argument('--apply_aug', default='cutout', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--id', default=1, type=int, help='number of repeated samples')
parser.add_argument('--recti_index', default=7, type=int, help='the rectification start time')

parser.add_argument('--weak_threshold', default=0.7, type=float)
parser.add_argument('--strong_threshold', default=0.97, type=float)
parser.add_argument('--apply_icc', action='store_true')
parser.add_argument('--num_negatives', default=512, type=int, help='number of negative keys')
parser.add_argument('--num_queries', default=256, type=int, help='number of queries per segment per image')
parser.add_argument('--consistency_rampup', default=300, type=int, help='number of queries per segment per image')

parser.add_argument('--temp', default=0.5, type=float)
parser.add_argument('--output_dim', default=64, type=int, help='output dimension from representation head')
parser.add_argument('--backbone', default='deeplabv3p', type=str, help='choose backbone: deeplabv3p, deeplabv2')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--binary_score_map_path', default=None, type=str)
parser.add_argument('--gt_pred_seg_image_path', default=None, type=str)
parser.add_argument('--mask_path', default=None, type=str)
parser.add_argument('--log_dir', default=None, type=str)
parser.add_argument('--snapshot_path', default='/home/wyy/PycharmProjects/ss/reco-main/saved/crack-1', type=str)
parser.add_argument('--Test', default=False, type=str)
parser.add_argument('--memco', default=0.5, type=float)
parser.add_argument('--alpha', default=0.6, type=float)



args = parser.parse_args()


##
random.seed(args.seed)
np.random.seed(args.seed) 
torch.manual_seed(args.seed)  
torch.cuda.manual_seed(args.seed) 
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.cuda.manual_seed_all(args.seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)


data_loader = BuildDataLoader(args.dataset, args.num_labels)
train_l_loader, train_u_loader, val_loader, test_loader = data_loader.build(supervised=False)

# Load Semantic Network
device = torch.device("cuda:{:d}".format(args.gpu) if torch.cuda.is_available() else "cpu")


if args.backbone == 'deeplabv3p':
    model = DeepLabv3Plus(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim,alpha=args.alpha).to(device)
elif args.backbone == 'deeplabv2':
    model = DeepLabv2(models.resnet101(pretrained=True), num_classes=data_loader.num_segments, output_dim=args.output_dim).to(device)
LPUR_model= LPUR(device,token_size=2, input_feature_dim=2048,
                                            feature_dim=2048, momentum=0.8,
                                            temperature=1, gumbel_rectification=False,co=args.memco).to(device)
total_epoch = 200
# RAMP_UP_ITERS=50
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)
optimizer_lpur = optim.SGD(LPUR_model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler_lpur = PolyLR(optimizer_lpur, total_epoch, power=0.9)
ema = EMA(model, 0.99)  # Mean teacher model


train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
val_epoch = len(val_loader)
avg_cost = np.zeros((total_epoch, 13))
iteration = 0


for index in range(total_epoch):
        cost = np.zeros(6)
        train_l_dataset = iter(train_l_loader)
        train_u_dataset = iter(train_u_loader)

        model.train()
        ema.model.train()
        LPUR_model.train()
        l_conf_mat = ConfMatrix(data_loader.num_segments)
        u_conf_mat = ConfMatrix(data_loader.num_segments)
        save_best = os.path.join(args.snapshot_path + '/model', 'model_best.pth')
        for i in range(train_epoch):
            train_l_data, train_l_label, _ = train_l_dataset.__next__()
            train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

            train_u_data, train_u_label, name = train_u_dataset.__next__()
            train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)
          
            optimizer.zero_grad()
            optimizer_lpur.zero_grad()

            # generate pseudo-labels
            with torch.no_grad():
                if index>args.recti_index:
                    LPUR_model.eval()
                    features = ema.model.encoder(train_u_data)
                    updated_query=LPUR_model(features[2],token_rectification=True)
                    pred_u,_=ema.model.decoder([features[0],features[1],updated_query])
                else:
                    pred_u,_= ema.model(train_u_data)
                pred_u_large_raw=pred_u
                unlab_ema_out_soft=torch.softmax(pred_u_large_raw, dim=1)
                pseudo_logits, pseudo_labels = torch.max(unlab_ema_out_soft, dim=1)

               
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                    data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

               
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=args.apply_aug)
            
                
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                    data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)
             

           
            pred_l, rep_ls,feature_le = model(train_l_data,tok_lern=True)
          
            pred_l_large=pred_l
          
            pred_u, rep_us= model(train_u_aug_data)
            pred_u_large=pred_u
           
               
            rep_l =[F.interpolate(rep_ls[i], size=pred_l.shape[-2:],mode='nearest') for i in range(len(rep_ls))]
            rep_u =[F.interpolate(rep_us[i], size=pred_l.shape[-2:],mode='nearest') for i in range(len(rep_us))]
           

            rep_l_m=torch.cat((rep_l[0],rep_l[1]),dim=1)
            rep_u_m=torch.cat((rep_u[0],rep_u[1]),dim=1)
           
            
            rep_all = torch.cat((rep_l_m, rep_u_m))

            pred_all = torch.cat((pred_l, pred_u))

            
            torch.use_deterministic_algorithms(False)
            sup_loss = compute_supervised_loss(pred_l_large, train_l_label)
           
           
            unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits,
                                                   args.strong_threshold)
            torch.use_deterministic_algorithms(True)

           
            if args.apply_icc:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(args.weak_threshold).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))
                   
                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))
                torch.use_deterministic_algorithms(False)
                icc_loss = compute_icc_loss(rep_all, label_all, mask_all, prob_all, args.strong_threshold,
                                              args.temp, args.num_queries, args.num_negatives)
                torch.use_deterministic_algorithms(True)
            else:
                icc_loss = torch.tensor(0.0)
              #prototype
            LPUR_model.train()
           
            learningloss=LPUR_model(feature_le[2],train_l_label,token_learning=True)
    
            loss = sup_loss + 0.5 * unsup_loss +0.5* icc_loss+0.4*learningloss[0]+0.2*learningloss[1]
            loss.backward()
            optimizer.step()
            ema.update(model)
            torch.use_deterministic_algorithms(False)
            l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())
            u_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), train_u_label.flatten())
            torch.use_deterministic_algorithms(True)
            
           
            freeze(model)
          
            features = model.encoder(train_l_data)
           
            updated_query=LPUR_model(features[2],token_rectification=True)
            
            pre_rec,_=model.decoder([features[0],features[1],updated_query])

            torch.use_deterministic_algorithms(False)
            distances = compute_supervised_loss(pre_rec, train_l_label)
            torch.use_deterministic_algorithms(True)
            distances.backward()
            optimizer_lpur.step()
            defreeze(model)

            cost[0] = sup_loss.item()
            cost[1] = unsup_loss.item()
            cost[2] = icc_loss.item()
            cost[3] = distances.item()
            cost[4] = learningloss[0].item()
            cost[5] = learningloss[1].item()

            avg_cost[index, :6] += cost / train_epoch
            iteration += 1

        avg_cost[index, 6:8] = l_conf_mat.get_metrics()
        avg_cost[index, 8:10] = u_conf_mat.get_metrics()
        scheduler.step()
        scheduler_lpur.step()
        with torch.no_grad():
            ema.model.eval()
            val_dataset = iter(val_loader)
            conf_mat = ConfMatrix(data_loader.num_segments)
            for i in range(val_epoch):
                test_data, test_label, _ = val_dataset.__next__()
                test_data, test_label = test_data.to(device), test_label.to(device)

                pred,_= ema.model(test_data)
                # pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
                torch.use_deterministic_algorithms(False)
                loss = compute_supervised_loss(pred, test_label)
                conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
                torch.use_deterministic_algorithms(True)
                avg_cost[index, 10] += loss.item() / test_epoch
            avg_cost[index, 11:] = conf_mat.get_metrics()

        print(
            'EPOCH: {:09d} ITER: {:09d} | TRAIN [Loss | mIoU | Acc.]: {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}  {:.9f} {:.9f}  || Test [Loss | mIoU | Acc.]: {:.9f} {:.9f} {:.9f}'
            .format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],
                    avg_cost[index][3], avg_cost[index][4], avg_cost[index][5], avg_cost[index][6], avg_cost[index][7],
                    avg_cost[index][8],
                    avg_cost[index][9],
                    avg_cost[index][10],
                    avg_cost[index][11],
                    avg_cost[index][12]))
        print('Top: mIoU {:.4f} Acc {:.4f}'.format(avg_cost[:, 11].max(), avg_cost[:, 12].max()))

     
        if avg_cost[index][11] >= avg_cost[:, 11].max():
            if args.apply_icc:
                torch.save(ema.model.state_dict(), save_best) 
            else:
                torch.save(ema.model.state_dict(), save_best)
               
        # if args.apply_icc:
        #     save_log = os.path.join(
        #         args.snapshot_path + '/logging/{}_label{}_semi_{}_reco_{}.npy'.format(args.dataset, args.num_labels,
        #                                                                               args.apply_aug, args.seed))
        #     np.save(save_log, avg_cost)
           
        # else:
        #     save_log = os.path.join(
        #         args.snapshot_path + '/logging/{}_label{}_semi_{}_{}.npy'.format(args.dataset, args.num_labels,
        #                                                                          args.apply_aug, args.seed))
        #     np.save(save_log, avg_cost)

          

            ######

        # wandb.log({"loss_sup": avg_cost[index][0],
        #            "epoch": index})
        # wandb.log({"loss_un": avg_cost[index][1],
        #            "epoch": index})
        # wandb.log({"loss_icc": avg_cost[index][2],
        #            "epoch": index})
        # wandb.log({"loss_center": avg_cost[index][3],
        #            "epoch": index})
        # wandb.log({"loss_dis": avg_cost[index][4],
        #            "epoch": index})
        # wandb.log({"loss_cla": avg_cost[index][5],
        #            "epoch": index})


        # wandb.log({"miou": avg_cost[index][11],
        #            "epoch": index})
        # wandb.log({"acc": avg_cost[index][12],
        #            "epoch": index})


if args.Test:
    results_log = open(args.log_dir, 'a')
    model = load_model(model, args.model_path)
   
    model.eval()
 
    total_batchtime=0
    num_batch=0
    with torch.no_grad():
      
        test_dataset = iter(test_loader)
        conf_mat = ConfMatrix(data_loader.num_segments)
        for i in range(test_epoch):
            test_data, test_label,names = test_dataset.__next__()
            test_data, test_label = test_data.to(device), test_label.to(device)
          
            pred,_= model(test_data)
          
           
            total_batchtime+=batch_time
            num_batch+=1
            # pred = F.interpolate(pred, size=test_label.shape[1:], mode='bilinear', align_corners=True)
            torch.use_deterministic_algorithms(False)
            loss = compute_supervised_loss(pred, test_label)
            conf_mat.update(pred.argmax(1).flatten(), test_label.flatten())
            torch.use_deterministic_algorithms(True)
            for j in range(pred.size()[0]):
                pr = F.softmax(pred[j].squeeze().permute(1, 2, 0), dim=-1).cpu().numpy()
                pr = pr.argmax(axis=-1)
                save_seg_results(pr, names[j], args.binary_score_map_path, args.gt_pred_seg_image_path, args.mask_path)
        
        mIoU,mAcc,iou,macro_precision, macro_recall, macro_f1= conf_mat.get_metrics_test()
        lines = []
        class_names = None
        for i in range(data_loader.num_segments):
            if class_names is None:
                cls = 'Class %d:' % (i + 1)
            else:
                cls = '%d %s' % (i + 1, class_names[i])
            lines.append('%-8s\t%.3f%%' % (cls, iou[i]))
        lines.append('----------------------------')
        lines.append('mIoU: {}%     mACC: {}%'.format(mIoU, mAcc))
        lines.append('precision: {}%    recall: {}%  f1 : {}% '.format(macro_precision, macro_recall,macro_f1))
        line = "\n".join(lines)

        results_log.write(line)
        results_log.write('\n')
        results_log.flush()
        results_log.close()

      
