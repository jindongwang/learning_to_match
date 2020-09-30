#torch=1.0.0
import argparse
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_loader
from IPython import embed
from model.L2M import L2M
from utils.utils_f1 import metric

def load_data():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    source_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    return source_loader, target_loader, test_loader

def evaluate(model, input_loader, calcf1=False):
    ori_train_state = model.is_train
    model.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        probabilities = model.predict(inputs)
        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)
    ret = {}
    probs, predict = torch.max(all_probs, 1)
    if calcf1:
        metr = metric(all_labels.cpu().detach().numpy(), predict.cpu().detach().numpy(), probs.cpu().detach().numpy())
        ret['metr'] = metr
    accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    model.set_train(ori_train_state)
    ret['accuracy'] = accuracy
    return ret

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
    assert (input_gnet != 0), 'GNet error!'
    gnet = GNet(input_gnet, [1024], 2)
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
    parser.add_argument('--match_feat_type', type=int, default=5,
                        help="""0: feature;
                                1: logits;
                                2: conditional loss + marginal loss;
                                3: feature + conditional loss + marginal loss;
                                4: logits + conditional loss + marginal loss
                                5: logits + feature + cond + marg loss (ALL)""")
    parser.add_argument('--init_lr', type=float, default=0.004)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--decay_rate', type=float, default=0.75)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--nesterov', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--cat_feature', type=str, default='column')
    parser.add_argument('--multi_gpu', type=str2bool,
                        nargs='?', const=True, default=False)
    parser.add_argument('--early_stop', type=int, default=50)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    class_num, width, srcweight, is_cen = get_data_config(args.dataset)
    assert (class_num != -1), 'Dataset name error!'
    assert (args.match_feat_type <= 5), 'option match_feat_type error!'

    if args.dataset == 'Covid-19':
        basenet='ResNet18'
        calcf1 = True
    else:
        basenet='ResNet50'
        calcf1 = False

    model = L2M(base_net=basenet, width=width, class_num=class_num, srcweight=srcweight, args=args)
    model.c_net.load_state_dict(torch.load(args.model_file))
    model.set_train(False)
    train_source_loader, train_target_loader, test_target_loader = load_data()
    ret = evaluate(model, test_target_loader, calcf1=calcf1)
    print('Test accuracy: ', ret['accuracy'])
    if calcf1:
        print('Test metric: ', ret['metr']) 
