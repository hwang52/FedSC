#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
from torch import eq
from torch.optim import SGD
from tqdm import tqdm
import copy, sys, math
import torch
import random
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from utils.options import args_parser
from model.resnet_base import resnet10
from utils.long_tailed_cifar10 import train_long_tail
from utils.dataset import classify_label, show_clients_data_distribution, Indices2Dataset
from utils.sample_dirichlet import clients_indices
from utils.util_func import average_kns
from utils.infonce_lcl import InfoNCE_LCL


model_dir = (Path(__file__).parent / "model").resolve()
if str(model_dir) not in sys.path: sys.path.insert(0, str(model_dir))
utils_dir = (Path(__file__).parent / "utils").resolve()
if str(utils_dir) not in sys.path: sys.path.insert(0, str(utils_dir))


class Global(object):
    def __init__(self, num_classes, device, args):
        self.device = device
        self.num_classes = num_classes
        self.args = args
        if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
            self.global_model = resnet10(nclasses=args.num_classes).to(args.device)
        else:
            exit('Load model error: unknown model!')

    def average_parameters(self, list_dicts_local_params, list_nums_local_data):
        global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            global_params[name_param] = value_global_param
        return global_params
    
    def get_dis_weights(self, list_local_num):
        list_local_dis = []
        global_num = [1/self.args.num_classes for i in range(self.args.num_classes)]
        for each_local_num in list_local_num:
            temp_add = 0.0
            each_local_num = [x / sum(each_local_num) for x in each_local_num]
            for i in range(len(each_local_num)):
                i_value = (each_local_num[i] - global_num[i]) ** 2
                temp_add += i_value
            temp_add = math.sqrt(temp_add * 0.5)
            list_local_dis.append(temp_add)
        return list_local_dis
    
    def get_co_p(self, re_p, list_local_dis, list_nums_local_data):
        list_e = []
        co_p = {}
        for temp_dis, temp_num in zip(list_local_dis, list_nums_local_data):
            t_value = temp_num/sum(list_nums_local_data) - temp_dis/sum(list_local_dis)
            t_value = 1 / (1 + np.exp(-t_value))
            list_e.append(t_value)
        list_e = list_e / sum(list_e)
        list_e = np.array(list_e)
        list_e = torch.from_numpy(list_e).to(self.args.device)
        for key, value in re_p.items():
            for i in range(len(re_p[key])):
                re_p[key][i] = re_p[key][i] * list_e[i]
        for key, value in re_p.items():
            co_p[key] = torch.sum(torch.stack(re_p[key]), dim=0)
        return co_p
    
    def get_re_p(self, list_local_p):
        re_p = {}
        for each_local_p in list_local_p:
            for key,value in each_local_p.items():
                if key in re_p:
                    re_p[key].append(value)
                else:
                    re_p[key] = [value]
        for key,value in re_p.items():
            temp = torch.stack(value)
            distances = torch.cdist(temp, temp, p=2)
            distances.fill_diagonal_(float("inf"))
            _, min_indexs = torch.min(distances, dim=1)
            for id,index in enumerate(min_indexs):
                temp[id] = (value[id] + value[index]) * 0.5
        return re_p

    def global_eval(self, average_params, data_test, batch_size_test):
        self.global_model.load_state_dict(average_params)
        self.global_model.eval()
        with torch.no_grad():
            num_corrects = 0
            test_loader = DataLoader(data_test, batch_size_test, shuffle=False)
            for data_batch in test_loader:
                images, labels = data_batch
                _, outputs = self.global_model(images.to(self.device))
                _, predicts = torch.max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def download_params(self):
        return self.global_model.state_dict()


class Local(object):
    def __init__(self, data_client, class_list, re_p, co_p):
        args = args_parser()
        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list
        if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
            self.local_model = resnet10(nclasses=args.num_classes).to(args.device)
        else:
            exit('Load model error: unknown model!')
        self.criterion = CrossEntropyLoss().to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)
        self.re_p = re_p
        self.co_p = co_p
        self.cl_tau = 0.05
        self.lcl_func = InfoNCE_LCL(self.cl_tau)
    
    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.cl_tau
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float).to(self.device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss

    def local_train(self, args, global_params, round_id):
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip()])
        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        local_p = {}
        client_class_num = []
        for tr_idx in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client, batch_size=args.batch_size_local_training, shuffle=True)
            local_class_num = [0 for i in range(args.num_classes)]
            for data_batch in data_loader:
                images, labels_g = data_batch
                images = images.to(self.device)
                labels = labels_g.to(self.device)
                if (args.data_name=='cifar10') or (args.data_name=='cifar100'):
                    images = transform_train(images)
                feas, outputs = self.local_model(images)
                for i in range(len(labels)):
                    local_class_num[labels_g[i].item()] += 1
                # Cross-Entropy loss
                ce_loss = self.criterion(outputs, labels)
                # neg and pos pairs
                if round_id > 1:
                    pos_key, neg_keys = [], []
                    reg_vector = []
                    for i in range(len(labels)):
                        label_id = labels_g[i].item()
                        try:
                            pos_key.append(torch.mean(self.re_p[label_id], dim=0))
                            reg_vector.append(self.co_p[label_id])
                        except:
                            pos_key.append(feas[i].detach())
                            reg_vector.append(feas[i].detach())
                        neg = [value for key, value in self.re_p.items() if key != label_id]
                        for each_neg_classes in neg: 
                            neg_keys.append(each_neg_classes)
                    # RPCL loss
                    rpcl_loss = self.lcl_func.infonce_lcl_loss(feas, pos_key, neg_keys)
                    # CPDR loss
                    cpdr_loss = torch.nn.functional.mse_loss(feas, torch.stack(reg_vector))
                    loss = ce_loss + rpcl_loss + cpdr_loss
                else:
                    loss = ce_loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # get client's prototypes
                if tr_idx == args.num_epochs_local_training - 1:  # last local epoch
                    for i in range(len(labels)):
                        if labels_g[i].item() in local_p:
                            local_p[labels_g[i].item()].append(feas[i,:])
                        else:
                            local_p[labels_g[i].item()] = [feas[i,:]]
            client_class_num = local_class_num
        local_p_avg = average_kns(local_p)
        return self.local_model.state_dict(), local_p_avg, client_class_num


def FedSC_main():
    args = args_parser()
    print(
        '=====> long-tail rate (imb_factor): {ib}\n'
        '=====> non-iid rate (non_iid): {non_iid}\n'
        '=====> activate clients (num_online_clients): {num_online_clients}\n'
        '=====> dataset classes (num_classes): {num_classes}\n'.format(
            ib=args.imb_factor,  # long-tail imbalance factor
            non_iid=args.non_iid_alpha,  # non-iid alpha based on Dirichlet-distribution
            num_online_clients=args.num_online_clients,  # activate clients in FL
            num_classes=args.num_classes,  # dataset classes
        )
    )
    random_state = np.random.RandomState(args.seed)
    # load dataset
    transform_c10 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]], 
                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    ])
    transform_c100 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343], 
                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    ])
    if args.data_name=='cifar10': 
        data_local_training = datasets.CIFAR10('./data/cifar10/', train=True, download=True, transform=transform_c10)
        data_global_test = datasets.CIFAR10('./data/cifar10/', train=False, transform=transform_c10)
    elif args.data_name=='cifar100':
        data_local_training = datasets.CIFAR100('./data/cifar100/', train=True, download=True, transform=transform_c100)
        data_global_test = datasets.CIFAR100('./data/cifar100/', train=False, transform=transform_c100)
    else:
        exit('Load dataset error: unknown dataset!')
    # distribute dataset
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)

    server_model = Global(num_classes=args.num_classes, device=args.device, args=args)
    total_clients = list(range(args.num_clients))
    indices2data = Indices2Dataset(data_local_training)
    fedsc_trained_acc = []
    re_p = {}
    co_p = {}
    for key in range(args.num_classes): # init
        re_p[key] = torch.empty(args.num_classes, device=args.device, requires_grad=False)
        co_p[key] = torch.empty(args.num_classes, device=args.device, requires_grad=False)

    # federated learning training
    for round_id in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = server_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []
        list_local_p, list_local_num = [], []
        # local model training
        for client in online_clients:
            indices2data.load(list_client2indices[client])
            data_client = indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client, class_list=original_dict_per_client[client], re_p=re_p, co_p=co_p)
            # local model update
            local_params, local_p, local_num = local_model.local_train(args, copy.deepcopy(global_params), round_id)
            list_dicts_local_params.append(copy.deepcopy(local_params))
            list_local_p.append(copy.deepcopy(local_p))
            list_local_num.append(local_num)
        re_p = server_model.get_re_p(list_local_p)
        dis_weights = server_model.get_dis_weights(list_local_num)
        co_p = server_model.get_co_p(re_p, dis_weights, list_nums_local_data)
        global_params = server_model.average_parameters(list_dicts_local_params, list_nums_local_data)
        one_re_train_acc = server_model.global_eval(global_params, data_global_test, args.batch_size_test)
        fedsc_trained_acc.append(one_re_train_acc)
        server_model.global_model.load_state_dict(copy.deepcopy(global_params))
        print("\nRound {} FedSC Accuracy: {}".format(round_id, fedsc_trained_acc))
    print("\n FedSC: ", fedsc_trained_acc)
    print("\n FedSC Top-1   Acc: ", max(fedsc_trained_acc))
    print("\n FedSC Average Acc: ", np.mean(fedsc_trained_acc))


if __name__ == '__main__':
    args = args_parser()
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    FedSC_main()
    # Example: python main_fedsc.py --data_name cifar10 --num_classes 10 --non_iid_alpha 0.05 --imb_factor 0.1 