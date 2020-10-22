# torch=1.0.0
import pretty_errors
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import torch
import torch.nn as nn
import numpy as np
import data_loader
from model import GNet, L2M, GNet2, L2MTrainerCritic
import datetime
import random
import os
import warnings
warnings.filterwarnings("ignore")


def pprint(*text):
    # print with UTC+8 time
    log_file = args.save_path.replace('.mdl', '.log')
    curr_time = '['+str(datetime.datetime.utcnow() +
                        datetime.timedelta(hours=8))[:19]+'] -'
    print(curr_time, *text, flush=True)
    if not os.path.exists(log_file):
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
        fp = open(log_file, 'w')
        fp.close()
    with open(log_file, 'a') as f:
        print(curr_time, *text, flush=True, file=f)


def load_data(root_path, source_dir, target_dir, batch_size):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    val_ratio = .2
    source_loader = data_loader.load_training(
        root_path, source_dir, batch_size, kwargs, train_val_split=val_ratio)
    target_loader = data_loader.load_training(
        root_path, target_dir, batch_size, kwargs, train_val_split=val_ratio)
    test_loader = data_loader.load_testing(
        root_path, target_dir, batch_size, kwargs)
    return source_loader, target_loader, test_loader


def get_data_config(dataset_name):
    class_num, width, srcweight, is_cen = -1, -1, -1, -1
    if dataset_name == 'Office-31':
        class_num = 31
        width = 2048
        srcweight = 3
        is_cen = False
    elif dataset_name == 'Office-Home':
        class_num = 65
        width = 2048
        srcweight = 3
        is_cen = False
    elif dataset_name == 'ImageCLEF-DA':
        class_num = 12
        width = 2048
        srcweight = 2
        is_cen = False
    elif dataset_name == 'VisDA':
        class_num = 12
        width = 2048
        srcweight = 3
        is_cen = False
    elif dataset_name == 'Covid-19':
        class_num = 2
        width = 512
        srcweight = 3
        is_cen = False
    return class_num, width, srcweight, is_cen


def init_gnet(width, class_num):
    input_gnet = 0
    if args.match_feat_type == 0:
        input_gnet = width
    elif args.match_feat_type == 1:
        input_gnet = class_num
    elif args.match_feat_type == 2:
        input_gnet = 2
    elif args.match_feat_type == 3:
        input_gnet = width + 2 if args.cat_feature == 'column' else width
    elif args.match_feat_type == 4:
        input_gnet = class_num + 2 if args.cat_feature == 'column' else class_num
    elif args.match_feat_type == 5:
        input_gnet = class_num + width + 2 if args.cat_feature == 'column' else class_num
    elif args.match_feat_type == 6:
        input_gnet = width + 1
    assert (input_gnet != 0), 'GNet error!'
    # gnet = GNet(input_gnet, [500, 300, 100], 1, use_set=False, drop_out=.5)
    gnet = GNet2(input_gnet, [256], 100, use_set=True, drop_out=.5)
    return gnet


def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--root_path', type=str, default="/home/jindwang/mine/data/office31",
                        help='the path to load the data')
    parser.add_argument('--source_dir', type=str, default="amazon",
                        help='the name of the source dir')
    parser.add_argument('--test_dir', type=str, default="webcam",
                        help='the name of the test dir')
    parser.add_argument('--save_path', type=str, default="AW.mdl",
                        help='the path to save the trained model')
    parser.add_argument('--save_folder', type=str, default='outputs')
    parser.add_argument('--use_adv', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--match_feat_type', type=int, default=0,
                        help="""0: feature;
                                1: logits;
                                2: conditional loss + marginal loss;
                                3: feature + conditional loss + marginal loss;
                                4: logits + conditional loss + marginal loss
                                5: logits + feature + cond + marg loss (ALL)""")
    parser.add_argument('--init_lr', type=float, default=5e-3)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--decay_rate', type=float, default=0.75)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--nesterov', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--cat_feature', type=str, default='column')
    parser.add_argument('--multi_gpu', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--glr', type=float, default=1e-2)
    parser.add_argument('--pretrain', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--exp', type=str, default='L2M')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    class_num, width, srcweight, is_cen = get_data_config(args.dataset)
    assert (class_num != -1), 'Dataset name error!'
    assert (args.match_feat_type <= 6), 'option match_feat_type error!'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.save_path = os.path.join(args.save_folder, args.save_path)
    args.log_file = args.save_path.replace('.mdl', '.log')

    pprint(args)

    gnet = init_gnet(1024, class_num)
    basenet = 'ResNet18' if args.dataset == 'Covid-19' else 'ResNet50'

    assist_model_old = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                           class_num=class_num, srcweight=srcweight, use_adv=args.use_adv, match_feat_type=args.match_feat_type, dataset=args.dataset, cat_feature=args.cat_feature)
    assist_model = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                       class_num=class_num, srcweight=srcweight, use_adv=args.use_adv, match_feat_type=args.match_feat_type, dataset=args.dataset, cat_feature=args.cat_feature)
    gnet = gnet.cuda()
    assist_model.c_net = assist_model.c_net.cuda()
    assist_model_old.c_net = assist_model_old.c_net.cuda()
    if args.multi_gpu:
        device_ids = [0, 1, 2, 3]
        gnet = torch.nn.DataParallel(gnet, device_ids)
        assist_model.c_net = torch.nn.DataParallel(
            assist_model.c_net, device_ids)
        assist_model_old.c_net = torch.nn.DataParallel(
            assist_model_old.c_net, device_ids)

    train_source_loader, train_target_loader, test_target_loader = load_data(
        args.root_path, args.source_dir, args.test_dir, args.batch_size)
    dataloaders = train_source_loader, train_target_loader, test_target_loader

    param_groups = assist_model.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    optimizer = torch.optim.SGD(param_groups,
                                lr=args.init_lr,
                                weight_decay=args.weight_decay,
                                momentum=.9
                                )

    trainer = L2MTrainerCritic(gnet, assist_model, assist_model_old,
                               optimizer, None, dataloaders, args)
    trainer.train()