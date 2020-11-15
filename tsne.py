import torch
import data_loader
import torch
import torch.nn as nn
import numpy as np
import data_loader
from model import L2M
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def tsne(model_file, data_folder):
    basenet = 'ResNet18'
    model = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                class_num=2, use_adv=True, match_feat_type=0)
    model = model.cuda()
    model.load_state_dict(torch.load(model_file))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    train_source_loader = data_loader.load_testing(
        data_folder, "pneumonia", 8, kwargs)
    train_target_loader = data_loader.load_testing(
        data_folder, "covid", 8, kwargs)
    first_inputs = True
    first_inputt = True
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
    total_feature = np.vstack((all_feats, all_featt))

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    total_only_tsne1 = tsne.fit_transform(total_feature)
    x_min, x_max = total_only_tsne1.min(0), total_only_tsne1.max(0)
    X_norm = (total_only_tsne1 - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 8))
    source_only_tsne = X_norm[:945, :]
    target_only_tsne = X_norm[945:, :]
    plt.scatter(source_only_tsne[:, 0],
                source_only_tsne[:, 1], s=4, c='g')
    plt.scatter(target_only_tsne[:, 0],
                target_only_tsne[:, 1], s=4, c='r')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('L2M.png')
    plt.show()


if __name__ == "__main__":
    model_file = "/home/jindwang/mine/code/learning_to_match/outputs/lamb-10-mu-0.01.mdl"
    data_folder = '/home/jindwang/mine/data/covid_folder/'

    tsne(model_file, data_folder)
