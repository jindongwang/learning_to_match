import torch
import torch.nn as nn
import os
import torchvision
import data_load
import backbone
import argparse
import pretty_errors
import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--nepoch', type=int, default=100)
    parser.add_argument('--basepath', type=str, default='/data/jindwang/data/all_data')
    args = parser.parse_args()
    return args

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyModel(nn.Module):
    def __init__(self, n_class=2, back='resnet18'):
        super(MyModel, self).__init__()
        basenet = backbone.network_dict[back]()
        fc = nn.Linear(basenet.output_num(), n_class)
        self.features = basenet
        self.fc = fc

    def forward(self, x):
        fea = self.features(x)
        pred = self.fc(fea)
        return pred
    


def train_net(model, dataloaders, optimizer):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.nepoch):
        total_loss = 0
        correct = 0
        model.train()
        for idx, (data, label) in enumerate(dataloaders['train']):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            correct += (predicted == label).sum().item()
            # if idx % 1 == 0:
            #     print('Loss: {:.6f}'.format(loss.item()))
        total_loss /= len(dataloaders['train'])
        train_acc = float(correct) * 1. / len(dataloaders['train'].dataset)
        valid_metr = test(model, dataloaders['val']) 
        test_metr = test(model, dataloaders['test'])
        print('[Epoch {}]: loss: {:.6f}, train acc: {:.4f}'.format(epoch, total_loss, train_acc))
        print(valid_metr)
        print(test_metr)
        print()

def test(model, dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    y_preds = torch.empty(0).cuda()
    y_trues = torch.empty(0).cuda()
    probs = torch.empty(0).cuda()
    with torch.no_grad():
        for _, (data, label) in enumerate(dataloader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            loss = criterion(pred, label)
            total_loss += loss.item()
            a, predicted = torch.max(pred.data, 1)
            correct += (predicted == label).sum().item()
            y_trues = torch.cat([y_trues.long(), label.long()])
            y_preds = torch.cat([y_preds.long(), predicted.long()])
            probs = torch.cat([probs.float(), a.float()])
        total_loss /= len(dataloader)
        metr = utils.metric(y_trues.cpu().detach().numpy(), y_preds.cpu().detach().numpy(), probs.cpu().detach().numpy())
        acc = float(correct) * 1. / len(dataloader.dataset)
    metr['acc'] = acc
    metr['loss'] = total_loss
    return metr
        

if __name__ == "__main__":
    torch.manual_seed(10)
    args = get_args()
    print(args)
    root_path = args.basepath
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    dataloaders = data_load.load_covid_data(root_path, batch_size=args.batchsize, kwargs=kwargs)
    print(len(dataloaders['train'].dataset))
    print(len(dataloaders['val'].dataset))
    print(len(dataloaders['test'].dataset))
 
    model = MyModel().to(device)
    optimizer = torch.optim.SGD(lr=args.lr, params=model.parameters())
    train_net(model, dataloaders, optimizer)
