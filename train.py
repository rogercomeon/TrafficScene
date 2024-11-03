import os
import oyaml as yaml
import time
import shutil
import torch
import random
import argparse
import numpy as np
from torch.nn.parallel.scatter_gather import gather
from torch.utils import data
from tqdm import tqdm
#from encoding.parallel import DataParallelModel, DataParallelCriterion
from model.pspnet_lga import pspnet_lga
from loader import get_loader
from utils import get_logger
from augmentations import get_composed_augmentations
import pdb
from metrics import runningScore, averageMeter
torch.backends.cudnn.enabled = True
       
def init_seed(manual_seed, en_cudnn=False):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = en_cudnn
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    random.seed(manual_seed)

def train(cfg, logger, logdir):
    # Setup seeds
    init_seed(11733, en_cudnn=False)       

    cfg["training"]["train_augmentations"]["scale"] = [540,720]
    cfg["training"]["train_augmentations"]["rcrop"] = [540,720]
    cfg["validating"]["val_augmentations"]["scale"] = [540,720]
    
    # Setup Augmentations
    train_augmentations = cfg["training"].get("train_augmentations", None) 
    t_data_aug = get_composed_augmentations(train_augmentations)    
    val_augmentations = cfg["validating"].get("val_augmentations", None)  
    v_data_aug = get_composed_augmentations(val_augmentations)

    data_loader = get_loader(cfg["data"]["dataset"])      
    data_path = '/data/roger/segementation'
    t_loader = data_loader(data_path,split=cfg["data"]["train_split"],augmentations=t_data_aug)
    v_loader = data_loader(data_path,split=cfg["data"]["val_split"],augmentations=v_data_aug)

    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg["training"]["batch_size"],
                                  num_workers=cfg["training"]["n_workers"],
                                  shuffle=True,
                                  drop_last=True  )
    valloader = data.DataLoader(v_loader,
                                batch_size=cfg["validating"]["batch_size"],
                                num_workers=cfg["validating"]["n_workers"] )

    logger.info("Using training seting {}".format(cfg["training"]))
    
    running_metrics_val1 = runningScore(t_loader.n_classes)
    running_metrics_val2 = runningScore(t_loader.n_classes)
    running_metrics_val3 = runningScore(t_loader.n_classes)
    running_metrics_val4 = runningScore(t_loader.n_classes)
    running_metrics_val5 = runningScore(t_loader.n_classes)
    running_metrics_val6 = runningScore(t_loader.n_classes)
    running_metrics_val7 = runningScore(t_loader.n_classes)
    running_metrics_val8 = runningScore(t_loader.n_classes)
    running_metrics_val9 = runningScore(t_loader.n_classes)
    # Setup Model and Loss          
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)     
    model = pspnet_lga(nclass=15, criterion=criterion,backbone=cfg["model"]["backbone"])
    modules_ori = [model.layer0,model.pretrained.layer1, model.pretrained.layer2, model.pretrained.layer3, model.pretrained.layer4,model.pretrained1.layer1,model.pretrained1.layer2,model.pretrained1.layer3,model.pretrained1.layer4,model.encoder1,model.SelfAttention]
    modules_new = [model.head1, model.auxlayer,model.head2,model.head3,model.head4,model.head5,model.head6,model.head7,model.head8,model.head9]

    if not args.pretrain is None: 
        pretrain_dict = torch.load(args.pretrain, map_location='cpu')
        state_dict = {'head.conv5.5.weight','head.conv5.5.bias','auxlayer.conv5.4.weight','auxlayer.conv5.4.bias'}  
        model_dict = {}
        for k, v in pretrain_dict.items():
            if k in state_dict:
                continue
            else:
                model_dict[k]=v
        model.load_state_dict(model_dict,strict=False)



    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=cfg["training"]["optimizer"]["lr0"]))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=cfg["training"]["optimizer"]["lr0"] * 10))
    optimizer = torch.optim.SGD(params_list, lr=cfg["training"]["optimizer"]["lr0"], momentum=cfg["training"]["optimizer"]["momentum"], weight_decay=cfg["training"]["optimizer"]["wd"])

    device_ids = [0,1]
    model = model.cuda()
    model = torch.nn.DataParallel(model,device_ids=device_ids)
    #Initialize training param
    cnt_iter = 0
    best_iou = 0.0
    best_iter = 0

    while cnt_iter <= cfg["training"]["train_iters"]:
        for (f_img, labels,img_name) in trainloader:
            cnt_iter += 1
            model.train()     
            f_img = [i.cuda() for i in f_img] 
            labels = [i.cuda() for i in labels]

            outputs,main_loss = model(f_img,labels)   
            loss = sum(main_loss)
            loss = torch.mean(loss)
            optimizer.zero_grad()  
            loss.backward()   
            optimizer.step()  
            current_lr = cfg["training"]["optimizer"]["lr0"] * (1-float(cnt_iter)/cfg["training"]["train_iters"]) ** cfg["training"]["optimizer"]["power"]            
            for index in range(0, len(modules_ori)):      
                optimizer.param_groups[index]['lr'] = current_lr
            for index in range(len(modules_ori), len(optimizer.param_groups)):    
                optimizer.param_groups[index]['lr'] = current_lr * 10
            
            if (cnt_iter + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f} "
                print_str = fmt_str.format( cnt_iter + 1,
                                            cfg["training"]["train_iters"],
                                            loss.item(), )   

                print(print_str)
                logger.info(print_str)
                
            if (cnt_iter + 1) % cfg["training"]["val_interval"] == 0 or (cnt_iter + 1) == cfg["training"]["train_iters"]:
                model.eval()
                with torch.no_grad():
                    confusion_matrix = np.zeros((15, 15))
                    for (f_img_val, labels_val,img_name_val) in tqdm(valloader):
                        f_img_val = [i.cuda() for i in f_img_val]
                        labels_val = [i.cuda() for i in labels_val]
                        outputs = model(f_img_val,labels_val)  
                        pred = [i.data.max(1)[1].cpu().numpy() for i in outputs]     
                        gt = [i.data.cpu().numpy() for i in labels_val]
                        running_metrics_val1.update(gt[0], pred[0])      
                        running_metrics_val2.update(gt[1], pred[1]) 
                        running_metrics_val3.update(gt[2], pred[2]) 
                        running_metrics_val4.update(gt[3], pred[3]) 
                        running_metrics_val5.update(gt[4], pred[4]) 
                        running_metrics_val6.update(gt[5], pred[5]) 
                        running_metrics_val7.update(gt[6], pred[6]) 
                        running_metrics_val8.update(gt[7], pred[7]) 
                        running_metrics_val9.update(gt[8], pred[8]) 
                        
                score1, class_iou1,class_acc1 = running_metrics_val1.get_scores()     
                for k, v in score1.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou1.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val1.reset()  

                score2, class_iou2,class_acc2 = running_metrics_val2.get_scores()     
                for k, v in score2.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))

                for k, v in class_iou2.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val2.reset()  

                score3, class_iou3,class_acc3 = running_metrics_val3.get_scores()     
                for k, v in score3.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou3.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val3.reset()  

                # 对第一个情景进行评估
                score4, class_iou4, class_acc4 = running_metrics_val4.get_scores()
                for k, v in score4.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou4.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val4.reset()

                # 对第二个情景进行评估
                score5, class_iou5, class_acc5 = running_metrics_val5.get_scores()
                for k, v in score5.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou5.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val5.reset()


                score6, class_iou6, class_acc6 = running_metrics_val6.get_scores()
                for k, v in score6.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou6.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val6.reset()


                score7, class_iou7, class_acc7 = running_metrics_val7.get_scores()
                for k, v in score7.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou7.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val7.reset()

                score8, class_iou8, class_acc8 = running_metrics_val8.get_scores()
                for k, v in score8.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou8.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val8.reset()


                score9, class_iou9, class_acc9 = running_metrics_val9.get_scores()
                for k, v in score9.items():     
                    print(k, v)
                    logger.info("{}: {}".format(k, v))
                for k, v in class_iou9.items(): 
                    logger.info("{}: {}".format(k, v))
                running_metrics_val9.reset()

                mean_iou= (score1["Mean IoU : \t"] + score2["Mean IoU : \t"] + score3["Mean IoU : \t"] + score4["Mean IoU : \t"] + score5["Mean IoU : \t"] + score6["Mean IoU : \t"] +score7["Mean IoU : \t"] + score8["Mean IoU : \t"] + score9["Mean IoU : \t"])/9
                if mean_iou > best_iou:  
                    best_iou = mean_iou
                    state = {
                        "epoch": cnt_iter + 1,
                        "model_state": model.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(logdir,
                        "{}_best_model_{}.pkl".format(cfg["model"]["arch"],cnt_iter+1),
                    )
                    torch.save(state, save_path)
                print("best iou:",best_iou)
    print("Final best iou:",best_iou)
#os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--pretrain",
        nargs="?",
        type=str,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Configuration file to use",
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    # pdb.set_trace()
    cfg["pretrain"] = args.pretrain
    cfg["data"]["dataset"] = args.dataset
    run_id = random.randint(1, 100000)
    if not os.path.exists("runs"):
        os.mkdir("runs")
    # logdir = os.path.join("runs",args.dataset+"_"+cfg["model"]["backbone"])     
    logdir = "/data/roger/allpic_segv1" 
    if not os.path.exists(logdir):
        os.mkdir(logdir)    
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")
    train(cfg, logger, logdir)
