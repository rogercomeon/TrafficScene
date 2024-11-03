import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import argparse
import timeit
import numpy as np
import oyaml as yaml
from torch.utils import data
import cv2
from model.pspnet_lga import pspnet_lga
from loader import get_loader
from metrics import runningScore
from utils import convert_state_dict
from augmentations import get_composed_augmentations
from collections import OrderedDict
from torchsummary import summary
import time
from thop import profile
import pdb
torch.backends.cudnn.benchmark = True

print(torch.__version__)

total_time = 0
def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    label_colours = [
                 # 0!=background
                 (112,176, 204), (32,   64,  0), (32, 64,  128),
                 (32, 128,  192), (81,  34, 215), (81,185,  10),
                 ( 192,96, 96), ( 96, 160, 160), ( 32,  224,  224),
                 (16,128,  64), (80,  128, 64), (240, 64, 64),
                 ( 187, 207,107), ( 80,32,128), (144,32,  192),]
    return label_colours

# 定义颜色映射表
color_mapping = {
    0: (204, 176, 112),
    1: (0, 64, 32),
    2: (128, 64, 32),
    3: (192, 128, 32),
    4: (215, 34, 81),
    5: (10, 185, 81),
    6: (96, 96, 192),
    7: (160, 160, 96),
    8: (224, 224, 32),
    9: (64, 128, 16),
    10: (64, 128, 80),
    11: (64, 64, 240),
    12: (107, 207, 187),
    13: (128, 32, 80),
    14: (192, 32, 144)
}
colors = np.array(list(color_mapping.values()))

# 创建output文件夹（如果不存在）
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def validate(cfg, args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg["training"]["train_augmentations"]["scale"] = [540,720]
    cfg["training"]["train_augmentations"]["rcrop"] = [540,720]
    cfg["validating"]["val_augmentations"]["scale"] = [540,720]
        
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    # Setup augmentations
    val_augmentations = cfg["validating"].get("val_augmentations", None)
    v_data_aug = get_composed_augmentations(val_augmentations)
    data_path = '/data/roger/segementation'

    v_loader = data_loader(data_path,split='test',augmentations=v_data_aug)
    
    n_classes = 15
    valloader = data.DataLoader(v_loader, batch_size=cfg["validating"]["batch_size"], num_workers=cfg["validating"]["n_workers"])

    running_metrics_val1 = runningScore(n_classes)
    running_metrics_val2 = runningScore(n_classes)
    running_metrics_val3 = runningScore(n_classes)
    running_metrics_val4 = runningScore(n_classes)
    running_metrics_val5 = runningScore(n_classes)
    running_metrics_val6 = runningScore(n_classes)
    running_metrics_val7 = runningScore(n_classes)
    running_metrics_val8 = runningScore(n_classes)
    running_metrics_val9 = runningScore(n_classes)

    # Setup Model
    model = pspnet_lga(nclass=15,backbone=cfg["model"]["backbone"]).to(device)

    state = torch.load(cfg["validating"]["resume"])["model_state"]  
    new_state = OrderedDict()
    for k,v in state.items():  
        name = k.replace("module.","")
        new_state[name] = v
    model.load_state_dict(new_state, strict=True)
    model.eval()
    model.to(device)
    print("Flip: ", cfg["validating"]["flip"])
    base_h = cfg["validating"]["base_size_h"]
    base_w = cfg["validating"]["base_size_w"]

    total = sum([param.nelement() for param in model.parameters()])
    
    print("Number of parameter: %.2fM" % (total/1e6))
    total_time = 0
    with torch.no_grad():
        pred = 0
        jishu = 0

        for (val, labels,img_name_val) in valloader:
            jishu = jishu + 1
            gt = [i.cuda() for i in labels]
            if cfg["validating"]["mult_scale"] == True:
                batch, _, ori_height, ori_width = val[0].size()
                assert batch == 1, "only supporting batchsize 1."  
                final_pred = torch.zeros([1, 15,ori_height,ori_width]).cuda()
                scales = [1]
                
                for scale in scales:     
                    val = [i.cuda() for i in val]
                    labels = [i.cuda() for i in labels]
                    if scale <= 1.0:   
                        
                        torch.cuda.synchronize()
                        start = time.time()
                        preds = model(val,gt)
                        torch.cuda.synchronize()
                        end = time.time()
                        total_time = total_time + end - start
                        if(jishu==100):
                            print(total_time)
                        pred = [i.data.max(1)[1].cpu().numpy() for i in preds] 
                    else:       
                        print(1)

            
            '''
            _val = val.to(device) 
            outputs = model(_val,gt)
            '''
             
            gt = [i.cpu().numpy() for i in labels]
            running_metrics_vals = [running_metrics_val1, running_metrics_val2, running_metrics_val3, running_metrics_val4, running_metrics_val5, running_metrics_val6, running_metrics_val7, running_metrics_val8, running_metrics_val9]
            loop_dir = os.path.join(output_dir,img_name_val[0]+'_pred')
            if not os.path.exists(loop_dir):
                os.makedirs(loop_dir)
            # 保存每个数组的可视化图像
            for i, array in enumerate(pred):
                # 将数组转换为NumPy数组
                # 使用颜色映射表进行索引，将像素值转换为RGB颜色
                image = colors[array]
                # 创建一个空的RGB图像
                image = np.zeros((540, 720, 3), dtype=np.uint8)

                # 将RGB颜色赋值给图像
                image[:, :] = colors[array].reshape(540,720,3)
                # 保存图像
                img_name = os.path.join(loop_dir, f'img{i}.png')
                # img_name_val = f'img_{i}_val.png'
                cv2.imwrite(img_name, image)

            loop_dir1 = os.path.join(output_dir,img_name_val[0]+'_gt')
            if not os.path.exists(loop_dir1):
                os.makedirs(loop_dir1)
            # 保存每个数组的可视化图像
            for i, array in enumerate(gt):
                # 将数组转换为NumPy数组
                # 使用颜色映射表进行索引，将像素值转换为RGB颜色
                image = colors[array]
                # 创建一个空的RGB图像
                image = np.zeros((540, 720, 3), dtype=np.uint8)
                # 将RGB颜色赋值给图像
                image[:, :] = colors[array].reshape(540,720,3)
                # 保存图像
                img_name = os.path.join(loop_dir1, f'img{i}.png')
                cv2.imwrite(img_name, image)
            for i, running_metrics_val in enumerate(running_metrics_vals):
                running_metrics_val.update(gt[i], pred[i])
                gt[i][gt[i]==255]=16

        score1, class_iou1,class_acc1 = running_metrics_val1.get_scores()     
        for k, v in score1.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou1[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc1[i])


        score2, class_iou2,class_acc2 = running_metrics_val2.get_scores()     
        for k, v in score2.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou2[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc2[i])


        score3, class_iou3,class_acc3 = running_metrics_val3.get_scores()     
        for k, v in score3.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou3[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc3[i])


        score4, class_iou4,class_acc4 = running_metrics_val4.get_scores()     
        for k, v in score4.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou4[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc4[i])


        score5, class_iou5,class_acc5 = running_metrics_val5.get_scores()     
        for k, v in score5.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou5[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc5[i])


        score6, class_iou6,class_acc6 = running_metrics_val6.get_scores()     
        for k, v in score6.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou6[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc6[i])


        score7, class_iou7,class_acc7 = running_metrics_val7.get_scores()     
        for k, v in score7.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou7[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc7[i])


        score8, class_iou8,class_acc8 = running_metrics_val8.get_scores()     
        for k, v in score8.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou8[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc8[i])


        score9, class_iou9,class_acc9 = running_metrics_val9.get_scores()     
        for k, v in score9.items():     
            print(k, v)
        print("each class iou")
        for i in range(n_classes):
            print(i, class_iou9[i])
        print("each class acc")
        for i in range(n_classes):
            print(i, class_acc9[i])

        mean_iou= (score1["Mean IoU : \t"] + score2["Mean IoU : \t"] + score3["Mean IoU : \t"] + \
                    score4["Mean IoU : \t"] + score5["Mean IoU : \t"] + score6["Mean IoU : \t"] + \
                    score7["Mean IoU : \t"] + score8["Mean IoU : \t"] + score9["Mean IoU : \t"])/9
        print("----------miou------------")
        print(mean_iou)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )       
    parser.add_argument(
        "--gpu",
        nargs="?",
        type=str,
        default="2",
        help="GPU ID",
    )      
    parser.add_argument(
        "--dataset",
        type=str,
        help="Configuration file to use",
    )      
    parser.add_argument("--save_dir",nargs="?",type=str,
            default="./output/",help="Path_to_Save",)
    parser.set_defaults(measure_time=True)
    args = parser.parse_args()
    with open(args.config) as fp:   
        cfg = yaml.safe_load(fp)
    cfg["data"]["dataset"] = args.dataset
    args.save_dir = 'output/'+args.dataset+'_'+cfg["model"]["backbone"]
    if not os.path.exists('output'):
        os.mkdir('output')
    if not os.path.exists('output/'+args.dataset):
        os.mkdir('output/'+args.dataset)
    if not os.path.exists(args.save_dir):       
        os.mkdir(args.save_dir)
    validate(cfg, args)
