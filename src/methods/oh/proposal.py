# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("Debugger attached")

import os.path as osp
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import transforms
from src.utils import loss, prompt_tuning, IID_losses
from src.models import network
from torch.utils.data import DataLoader
from src.data.data_list import ImageList_idx, ImageList_idx_aug_fix
from sklearn.metrics import confusion_matrix
# from clip.custom_clip import get_coop
# import clip
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from clip.ReCLIP import ReCLIP
from src.utils.utils import *
logger = logging.getLogger(__name__)

def data_load(cfg): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = cfg.TEST.BATCH_SIZE
    txt_tar = open(cfg.t_dset_path).readlines()
    txt_test = open(cfg.test_dset_path).readlines()
    if not cfg.DA == 'uda':
        label_map_s = {}
        for i in range(len(cfg.src_classes)):
            label_map_s[cfg.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in cfg.tar_classes:
                if int(reci[1]) in cfg.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()
    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=cfg.NUM_WORKERS, drop_last=False)
    dsets["source"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=False)
    return dset_loaders

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
    #normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  #else:
   # normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def print_cfg(cfg):
    s = "==========================================\n"
    for arg, content in cfg.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(cfg):
    # text_inputs = clo    
    # model = 
    dset_loaders = data_load(cfg)
    num_sample=len(dset_loaders["target"].dataset)
    # num_sample=len(dset_loaders["target"].dataset)

    model = ReCLIP(cfg, num_sample)

    if cfg.MODEL.ARCH[0:3] == "res":
        netF = network.ResBase(res_name = cfg.MODEL.ARCH).cuda()
    elif cfg.MODEL.ARCH[0:3] == "vgg":
        netF = network.VGGBase(vgg_name = cfg.MODEL.ARCH).cuda()
    netB = network.feat_bottleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=cfg.bottleneck).cuda()
    netC = network.feat_classifier(type='wn', class_num = cfg.class_num, bottleneck_dim=cfg.bottleneck).cuda()

    modelpath = cfg.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = cfg.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_ib = []

    for k, v in netF.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if cfg.OPTIM.LR_DECAY2 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if cfg.OPTIM.LR_DECAY1 > 0:
            param_group += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]
        else:
            v.requires_grad = False
    # for k, v in model.prompt_learner.named_parameters():
    #     if(v.requires_grad == True):
    #         param_group_ib += [{'params': v, 'lr': cfg.OPTIM.LR * cfg.OPTIM.LR_DECAY1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # optimizer_ib = optim.SGD(param_group_ib)
    # optimizer_ib = op_copy(optimizer_ib)
    # optim_state = deepcopy(optimizer_ib.state_dict())
    # model.reset_classnames(cfg.classname, cfg.ProDe.ARCH)
    max_iter = cfg.TEST.MAX_EPOCH * len(dset_loaders["target"])
    interval_iter = max_iter // cfg.TEST.INTERVAL
    iter_num = 0
    logtis_bank = torch.randn(num_sample, cfg.class_num).cuda()

    loader = dset_loaders["source"]
    # print(len(loader.dataset))
    # print(num_sample)
    # with torch.no_grad():
    #     iter_test = iter(loader)
    #     for i in range(len(loader)):
    #         data = next(iter_test)
    #         inputs = data[0]
    #         indx = data[-1]
    #         # print(indx)
    #         inputs = inputs.cuda()
    #         output = netB(netF(inputs))
    #         outputs = netC(output)
    #         outputs=nn.Softmax(dim=1)(outputs)
    #         logtis_bank[indx] = outputs.detach().clone()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            indx=data[-1]
            inputs = inputs.cuda()
            output = netB(netF(inputs))
            outputs = netC(output)
            outputs=nn.Softmax(dim=1)(outputs)
            logtis_bank[indx] = outputs.detach().clone()
            # logtis_bank = logtis_bank.cpu()

    test_loader = dset_loaders["target"]
    while iter_num < max_iter:
        try:
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter(test_loader))
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test, inputs_test_augs), _, tar_idx = next(iter_test)
        
        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0:
            _, pseudo_labels, logits = model.generate_pseudo_labels(test_loader)
            model.update_ReCLIP_modules(test_loader)
        
        inputs_test = inputs_test.cuda()
        inputs_test_augs = inputs_test_augs[0].cuda()

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

    # for epoch in range(cfg.TEST.MAX_EPOCH):
    #     pseudo_labels = model.generate_pseudo_labels(test_loader)
    #     model.update_ReCLIP_modules(test_loader)

    #     for (inputs_test, inputs_test_augs), _, tar_idx in test_loader:
    #         if torch.max(tar_idx) >= logtis_bank.size(0):
    #              raise ValueError(f"tar_idx {torch.max(tar_idx).item()} >= logtis_bank size {logtis_bank.size(0)}")
    #         # print(tar_idx)
    #         # break
    #         inputs_test = inputs_test.cuda()
    #         inputs_test_augs = inputs_test_augs[0].cuda() 

    #         iter_num += 1
    #         lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        with torch.no_grad():
            outputs_test_new = outputs_test.clone().detach()
            
        if cfg.PROPOSAL.ARCH == "RN50":
            inputs_test_clip = inputs_test_augs
        else:
            inputs_test_clip = inputs_test
        
        clip_score = torch.from_numpy(logits).cuda() if isinstance(logits, np.ndarray) else logits.cuda()
        # clip_score = clip_score.float()
        # print(clip_score.shape)
        with torch.no_grad():
            new_clip = (outputs_test_new - logtis_bank[tar_idx].cuda()) + clip_score[tar_idx]
            clip_score_sm = nn.Softmax(dim = 1)(new_clip)
            _,clip_index_new = torch.max(new_clip, 1)
        iic_loss = IID_losses.IID_loss(softmax_out, clip_score_sm)
        classifier_loss = cfg.PROPOSAL.IIC_PAR * iic_loss
        msoftmax = softmax_out.mean(dim = 0)

        if  cfg.SETTING.DATASET=='office':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.PROPOSAL.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='office-home':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.PROPOSAL.EPSILON))
            classifier_loss = classifier_loss - 1.0 * gentropy_loss
        if  cfg.SETTING.DATASET=='VISDA-C':
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + cfg.PROPOSAL.EPSILON))
            classifier_loss = classifier_loss - 0.1 * gentropy_loss
        pred = clip_index_new
        entropy_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        classifier_loss =  cfg.PROPOSAL.GENT_PAR*entropy_loss + classifier_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            if cfg.SETTING.DATASET=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%;loss ={}'.format(cfg.name, iter_num, max_iter, acc_s_te,classifier_loss)

            logger.info(log_str)
            netF.train()
            netB.train()
            netC.train()
    if cfg.ISSAVE:   
        torch.save(netF.state_dict(), osp.join(cfg.output_dir, "target_F_" + cfg.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(cfg.output_dir, "target_B_" + cfg.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(cfg.output_dir, "target_C_" + cfg.savename + ".pt"))
        
    return netF, netB, netC