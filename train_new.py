import pretty_errors
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
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
    train_val_split = -1
    source_loader = data_loader.load_training(
        root_path, source_dir, batch_size, kwargs, train_val_split)
    target_loader = data_loader.load_training(
        root_path, target_dir, batch_size, kwargs, train_val_split)
    test_loader = data_loader.load_testing(
        root_path, target_dir, batch_size, kwargs)
    return source_loader, target_loader, test_loader


def get_data_config(dataset_name):
    class_num, width, srcweight, is_cen = -1, -1, -1, -1
    if dataset_name.lower() in ['office-31', 'office31', 'o31']:
        class_num = 31
        width = 2048
        srcweight = 3
        is_cen = False
    elif dataset_name.lower() in ['office-home', 'officehome', 'ohome']:
        class_num = 65
        width = 2048
        srcweight = 3
        is_cen = False
    elif dataset_name.lower() in ['imageclef-da', 'clef', 'imageclef', 'imageclefda']:
        class_num = 12
        width = 2048
        srcweight = 2
        is_cen = False
    elif dataset_name.lower() in ['visda', 'visda17', 'visda-17']:
        class_num = 12
        width = 2048
        srcweight = 3
        is_cen = False
        args.source_dir = 'train'
        args.test_dir = 'validation'
    elif dataset_name.lower() in ['covid-19', 'covid', 'covid19']:
        class_num = 2
        width = 512
        srcweight = 3
        is_cen = False
        args.source_dir = 'pneumonia'
        args.test_dir = 'covid'
        args.save_path = 'covid.mdl'
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
    gnet = GNetGram(args.batch_size ** 2, [512, 256], 1, use_set=True, drop_out=.5, mono=True, init_net=False)
    # gnet = GNet(input_gnet, [512, 256], 1, use_set=True, drop_out=.5, mono=False, init_net=True)
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', default='Office-Home', type=str,
                        help='which dataset')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--root_path', type=str, default="/home/jindwang/mine/data/OfficeHome",
                        help='the path to load the data')
    parser.add_argument('--source_dir', type=str, default="Art",
                        help='the name of the source dir')
    parser.add_argument('--test_dir', type=str, default="Clipart",
                        help='the name of the test dir')
    parser.add_argument('--save_path', type=str, default="AC.mdl",
                        help='the path to save the trained model')
    parser.add_argument('--save_folder', type=str, default='outputs')
    parser.add_argument('--use_adv', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--match_feat_type', type=int, default=0, choices=[0,1,2,3,4,5],
                        help="""0: feature;
                                1: logits;
                                2: conditional loss + marginal loss;
                                3: feature + conditional loss + marginal loss;
                                4: logits + conditional loss + marginal loss
                                5: logits + feature + cond + marg loss (ALL)""")
    parser.add_argument('--init_lr', type=float, default=0.004)
    parser.add_argument('--glr', type=float, default=0.0005,
                        help='learning rate for gnet')
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--decay_rate', type=float, default=0.75)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--nesterov', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--cat_feature', type=str, default='column')
    parser.add_argument('--multi_gpu', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--exp', type=str, default='l2m')
    parser.add_argument('--gopt', type=str, default='sgd')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    class_num, width, srcweight, is_cen = get_data_config(args.dataset)
    assert (class_num != -1), 'Dataset name error!'
    assert (args.match_feat_type <= 6), 'option match_feat_type error!'

    # controls random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.save_path = os.path.join(args.save_folder, args.save_path)
    args.log_file = args.save_path.replace('.mdl', '.log')

    pprint(vars(args))

    gnet = init_gnet(1024, class_num)
    basenet = 'ResNet18' if args.dataset == 'Covid-19' else 'ResNet50'

    model_old = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                           class_num=class_num, srcweight=srcweight, use_adv=args.use_adv, match_feat_type=args.match_feat_type, dataset=args.dataset, cat_feature=args.cat_feature)
    model = L2M(base_net=basenet, bottleneck_dim=1024, width=256,
                       class_num=class_num, srcweight=srcweight, use_adv=args.use_adv, match_feat_type=args.match_feat_type, dataset=args.dataset, cat_feature=args.cat_feature)
    gnet = gnet.cuda()
    model.net = model.net.cuda()
    model_old.net = model_old.net.cuda()

    # controls multi-gpu training
    if args.multi_gpu:
        device_ids = [0, 1]
        gnet = torch.nn.DataParallel(gnet, device_ids)
        model.net = torch.nn.DataParallel(
            model.net, device_ids)
        model_old.net = torch.nn.DataParallel(
            model_old.net, device_ids)

    train_source_loader, train_target_loader, test_target_loader = load_data(
        args.root_path, args.source_dir, args.test_dir, args.batch_size)
    dataloaders = train_source_loader, train_target_loader, test_target_loader

    trainer = L2MTrainer(gnet, model, model_old, dataloaders, args)
    trainer.train()
