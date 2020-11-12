import torch
import data_loader
import argparse
import torch
import torch.nn as nn
import numpy as np
import data_loader
from model import GNet, L2M, L2MTrainer, GNetGram
import datetime
import random
import os
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def tsne():
    basenet = 'ResNet18'
    model = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                       class_num=2, use_adv=True, match_feat_type=0)
    model = model.cuda()
    #model.net=torch.load("lamb-10-mu-0.01.mdl",map_location='cpu')
    model.load_state_dict(torch.load("/home/jindwang/mine/code/learning_to_match/outputs/lamb-10-mu-0.01.mdl"))
    #model=torch.jit.load("lamb-10-mu-0.01.mdl",map_location='cpu')
    
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_val_split = -1
    train_source_loader = data_loader.load_training("/home/jindwang/mine/data/covid_folder/", "pneumonia", 8, kwargs, -1)
    train_target_loader = data_loader.load_training("/home/jindwang/mine/data/covid_folder/", "covid", 8, kwargs, -1)
    first_inputs=True
    first_inputt=True
    model.eval()
    with torch.no_grad():
        for datas, datat in zip(train_source_loader, train_target_loader):
            if len(datas) != len(datat):
                continue
            inputs_source, labels = datas
            inputs_target, labelt = datat
            inputs_source, labels = inputs_source.cuda(), labels.cuda()
            inputs_target, labelt = inputs_target.cuda(), labelt.cuda()
            inputs = torch.cat((inputs_source, inputs_target), dim=0)
            #inputs_source, labels = inputs_source.cuda(), l, dim=0abels.cuda()
            #inputs_target, labelt = inputs_target.cuda(), labelt.cuda()
            # feats, __, __, __, __ = model(inputs_source)
            # featt, __, __, __, __ = model(inputs_target)
            feat, _, _, _, _, _, _, _ = model(
                    inputs, labels)
            feats, featt = feat[:feat.size(0) // 2], feat[feat.size(0) // 2:]
            if first_inputs:
                all_feats = feats
                first_inputs = False
            else:
                all_feats = torch.cat((all_feats, feats), 0)
            if first_inputt:
                all_featt = featt
                first_inputt = False
            else:
                all_featt = torch.cat((all_featt, featt), 0)
    all_feats = all_feats.cpu().detach().numpy()
    all_featt = all_featt.cpu().detach().numpy()
    total_feature1 = np.vstack((all_feats, all_featt))
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    total_only_tsne1 = tsne.fit_transform(total_feature1)
    x_min, x_max = total_only_tsne1.min(0), total_only_tsne1.max(0)
    X_norm = (total_only_tsne1 - x_min) / (x_max - x_min)
                
    plt.figure(figsize=(8,8))
    source_only_tsne = X_norm[:945,:]
    target_only_tsne = X_norm[945:,:]
    np.save('all_DANN_feat_source',total_only_tsne1[:945,:])
    np.save('all_DANN_feat_target',total_only_tsne1[945:,:])
    plt.scatter(source_only_tsne[:,0], source_only_tsne[:,1], s=4, c='g', alpha=0.4)
    plt.scatter(target_only_tsne[:,0], target_only_tsne[:,1], s=4, c='r', alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('L2M.png')
    plt.show()


if __name__ == "__main__":
    tsne()
