import pretty_errors
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import data_loader
from model import GNet, L2MNet, L2MTrainer, GNetGram, GNetTransformer
import datetime
from utils import helper
import os
import warnings
warnings.filterwarnings("ignore")


def pprint(*text):
    # print with UTC+8 time
    log_file = args.save_path.replace('.pkl', '.log')
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
    elif dataset_name.lower() in ['office-home', 'officehome', 'ohome']:
        class_num = 65
        width = 2048
    elif dataset_name.lower() in ['imageclef-da', 'clef', 'imageclef', 'imageclefda']:
        class_num = 12
        width = 2048
    elif dataset_name.lower() in ['visda', 'visda17', 'visda-17']:
        class_num = 12
        width = 2048
        args.save_path = 'visda.pkl'
    elif dataset_name.lower() in ['covid-19', 'covid', 'covid19']:
        class_num = 2
        width = 512
        args.save_path = 'covid.pkl'
    elif dataset_name.lower() in ['visda-binary', 'vbinary']:
        class_num = 2
        width = 2048
        args.save_path = 'visda-binary.pkl'
    elif dataset_name.lower() in ['bac']:
        class_num = 2
        width = 512
        args.save_path = 'bac.pkl'
    elif dataset_name.lower() in ['viral']:
        class_num = 2
        width = 512
        args.save_path = 'viral.pkl'
    return class_num, width, srcweight, is_cen


def init_gnet(g_input):
    # gnet = GNetGram(2 * args.meta_m ** 2, [256, 128], 1,
    #                 use_set=True, drop_out=.5, mono=False, init_net=True)
    gnet = GNetGram(args.gbatch ** 2, [128, 64], 1,
                    use_set=True, drop_out=.5, mono=False, init_net=True)
    # gnet = GNet(input_gnet, [512, 256], 1, use_set=True,
    #             drop_out=.5, mono=False, init_net=True)
    # gnet = GNetTransformer(32 * 32, [128, 64], 1,
    #                 use_set=True, drop_out=.5, mono=False, init_net=True)
    return gnet


def get_args():
    parser = argparse.ArgumentParser()

    # training setting
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--dataset', default='ohome', type=str,
                        help='which dataset')
    parser.add_argument('--seed', type=int, default=52, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--src', type=str, default="pneumonia",
                        help='the name of the source dir')
    parser.add_argument('--tar', type=str, default="covid",
                        help='the name of the test dir')
    parser.add_argument('--use_adv', action='store_true', default=False)
    parser.add_argument('--init_lr', type=float, default=0.004)
    parser.add_argument('--glr', type=float, default=0.0005,
                        help='learning rate for gnet')
    parser.add_argument('--gamma', type=float, default=10)
    parser.add_argument('--decay_rate', type=float, default=0.75)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--nesterov', action='store_false', default=True)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--early_stop', type=int, default=20)

    parser.add_argument('--gopt', type=str, default='sgd')
    parser.add_argument('--meta_m', type=int, default=8)
    parser.add_argument('--gbatch', type=int, default=16)
    parser.add_argument('--lamb', type=float, default=10)
    parser.add_argument('--mu', type=float, default=0.01)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_model_file', type=str, default='model.pkl')

    # save, path, folder
    parser.add_argument('--data_path', type=str, default="/home/jindwang/mine/data/covid_folder/",
                        help='the path to load the data')
    parser.add_argument('--save_path', type=str, default="model.pkl",
                        help='the path to save the trained model')
    parser.add_argument('--save_folder', type=str, default='outputs', help='results save folder')
    parser.add_argument('--exp', type=str, default='l2m', help='Experiment name')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    class_num, width, srcweight, is_cen = get_data_config(args.dataset)
    args.save_path = f"{args.dataset}-lamb-{args.lamb}-mu-{args.mu}.pkl"
    assert (class_num != -1), 'Dataset name error!'

    # helper.set_random(args.seed)
    args.save_path = os.path.join(args.save_folder, args.save_path)
    args.log_file = args.save_path.replace('.pkl', '.log')

    pprint(vars(args))

    gnet = init_gnet(args.gbatch ** 2)
    basenet = 'ResNet18' if args.dataset.lower(
    ) in ['covid-19', 'covid', 'covid19', 'bac', 'viral'] else 'ResNet50'

    model_old = L2MNet(base_net=basenet, bottleneck_dim=1024, width=256,
                    class_num=class_num, use_adv=args.use_adv)
    model = L2MNet(base_net=basenet, bottleneck_dim=1024, width=256, class_num=class_num,
                use_adv=args.use_adv)

    gnet = gnet.cuda()
    model = model.cuda()
    model_old = model_old.cuda()

    # controls multi-gpu training
    if args.multi_gpu:
        device_ids = [0, 1]
        gnet = torch.nn.DataParallel(gnet, device_ids)
        model.net = torch.nn.DataParallel(model, device_ids)
        model_old.net = torch.nn.DataParallel(model_old, device_ids)

    train_source_loader, train_target_loader, test_target_loader = load_data(
        args.data_path, args.src, args.tar, args.batch_size)
    dataloaders = train_source_loader, train_target_loader, test_target_loader

    trainer = L2MTrainer(gnet, model, model_old, dataloaders, args)
    if not args.test:
        trainer.train()
    else:
        test_path = os.path.join('/home/jindwang/mine/code/learning_to_match/outputs', args.test_model_file)
        model.load_state_dict(torch.load(test_path))
        calcf1 = True if args.dataset.lower() in ['covid-19', 'covid', 'covid19', 'visda', 'bac', 'viral'] else False
        ret = trainer.evaluate(model, test_target_loader, calcf1=calcf1)
        print(trainer.inference(model, test_target_loader))
        print('Test result:')
        print(ret)
        

        # import os
        # import PIL
        # import numpy as np
        # import torch
        # import torch.nn.functional as F
        # import torchvision.models as models
        # from torchvision.utils import make_grid, save_image
        # import glob

        # from gradcam_vis import visualize_cam, Normalize, GradCAM, GradCAMpp
        # img_dir = '/home/jindwang/mine/data/covid_folder/covid/0/'
        # for file in glob.glob(os.path.join(img_dir, '*')):
        #     print(file)
        #     img_path = file
        #     img_name = img_path.split('/')[-1]

        #     pil_img = PIL.Image.open(img_path)
        #     pil_img = pil_img.convert('RGB')
        #     normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #     torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
        #     torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
        #     normed_torch_img = normalizer(torch_img)

        #     resnet = models.resnet18(pretrained=False)
        #     model = model.cpu()
        #     resnet.conv1 = model.base_network.conv1
        #     resnet.bn1 = model.base_network.bn1
        #     resnet.relu = model.base_network.relu
        #     resnet.maxpool = model.base_network.maxpool
        #     resnet.layer1 = model.base_network.layer1
        #     resnet.layer2 = model.base_network.layer2
        #     resnet.layer3 = model.base_network.layer3
        #     resnet.layer4 = model.base_network.layer4
        #     resnet.avgpool = model.base_network.avgpool
        #     resnet.eval()
        #     cam_dict = dict()
        #     resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        #     resnet_gradcam = GradCAM(resnet_model_dict, True)
        #     resnet_gradcampp = GradCAMpp(resnet_model_dict, True)
        #     cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

        #     images = []
        #     for gradcam, gradcam_pp in cam_dict.values():
        #         mask, _ = gradcam(normed_torch_img)
        #         heatmap, result = visualize_cam(mask.cpu(), torch_img)

        #         mask_pp, _ = gradcam_pp(normed_torch_img)
        #         heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
                
        #         # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
        #         images.append(torch.stack([result], 0))
                
        #     images = make_grid(torch.cat(images, 0), nrow=5)

        #     output_dir = '/home/jindwang/mine/code/learning_to_match/outputs-vis/second-normal'
        #     if not os.path.exists(output_dir):
        #         os.mkdir(output_dir)
        #     output_name = img_name
        #     output_path = os.path.join(output_dir, output_name)
        #     save_image(images, output_path)
            # PIL.Image.open(output_path)

