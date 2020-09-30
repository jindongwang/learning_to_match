import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
from utils import mmd


class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class PositiveLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight**2, self.bias)


class MLP(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, drop_out=0, mono=False):
        super(MLP, self).__init__()
        hidden = []
        for layer in n_hiddens:
            if mono:
                hidden.append(PositiveLinear(n_input, layer))
            else:
                hidden.append(nn.Linear(n_input, layer))
            hidden.append(nn.ReLU(inplace=True))
            if drop_out != 0:
                hidden.append(nn.Dropout(drop_out))
            n_input = layer
        self.hidden = nn.Sequential(*hidden)
        if mono:
            self.out = PositiveLinear(n_input, n_output)
        else:
            self.out = nn.Linear(n_input, n_output)
        self.init_layers(mono=mono)

    def forward(self, x):
        x = self.hidden(x)
        x = self.out(x)
        return x

    def init_layers(self, weight=0.01, bias=0, mono=False):
        for layer in self.hidden:
            if mono:
                if isinstance(layer, PositiveLinear):
                    layer.weight.data.normal_(0, weight)
                    layer.bias.data.fill_(bias)
            else:
                if isinstance(layer, nn.Linear):
                    layer.weight.data.normal_(0, weight)
                    layer.bias.data.fill_(bias)
        self.out.weight.data.normal_(0, weight)
        self.out.bias.data.fill_(bias)


class GNet(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False):
        super(GNet, self).__init__()
        net = MLP(n_input, n_hiddens, n_output, drop_out, mono=mono)
        self.hidden = net.hidden
        self.out = net.out
        self.set = use_set
        self.parameter_list = [{"params": self.hidden.parameters(), "lr": 0.1}, {
            "params": self.out.parameters(), "lr": 0.1}]

    def forward(self, x):
        x = self.hidden(x)
        if self.set:
            x_s = torch.sum(x[:x.size(0) // 2], dim=0, keepdim=True)
            x_t = torch.sum(x[x.size(0) // 2:], dim=0, keepdim=True)
            x = torch.cat((x_s, x_t), dim=0)
        x = self.out(x)
        out = torch.sigmoid(x)
        return out


class GNet2(nn.Module):
    def __init__(self, n_input, n_hiddens, n_output, use_set=True, drop_out=0, mono=False):
        super(GNet2, self).__init__()
        self.use_set = use_set
        self.net_s = MLP(n_input, n_hiddens, n_output,
                         drop_out=drop_out, mono=mono)
        self.net_t = MLP(n_input, n_hiddens, n_output,
                         drop_out=drop_out, mono=mono)
        self.combine = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_out),
            nn.Linear(256, 1)
        )

    def forward(self, x_src, x_tar):
        # fea_src, fea_tar = F.relu(self.net_s(x_src)), F.relu(self.net_s(x_tar))
        fea_src, fea_tar = x_src, x_tar
        if self.use_set:
            fea_src = torch.mean(fea_src, dim=0, keepdim=True)
            fea_tar = torch.mean(fea_tar, dim=0, keepdim=True)
        feas = fea_src - fea_tar
        out = self.combine(feas)
        # out = out * out.t()
        # out = F.relu(out)
        # out = torch.sigmoid(out)
        return out


class L2MNet(nn.Module):
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=31, use_adv=True):
        super(L2MNet, self).__init__()
        self.use_adv = use_adv
        # set base network
        self.base_network = backbone.network_dict[base_net]()
        self.bottleneck_layer = nn.Sequential(*[nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                                nn.BatchNorm1d(bottleneck_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5)])
        self.classifier_layer = nn.Sequential(*[nn.Linear(bottleneck_dim, width),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(width, class_num)])
        self.softmax = nn.Softmax(dim=1)
        # initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1},
                               {"params": self.bottleneck_layer.parameters(),
                                "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1}]

        if self.use_adv:
            self.grl_layer = GradientReverseLayer()
            self.classifier_layer_2 = MLP(
                bottleneck_dim, [width], class_num, drop_out=.5)
            # DANN: add DANN, make L2M not only class-invariant (as conditional loss) but also domain invariant (as marginal loss)
            self.domain_classifier = MLP(
                bottleneck_dim, [bottleneck_dim, width], 2, drop_out=.5)

            self.parameter_list.append(
                {"params": self.classifier_layer_2.parameters(), "lr": 1})
            self.parameter_list.append(
                {"params": self.domain_classifier.parameters(), "lr": 1})

    def forward(self, inputs):
        features = self.base_network(inputs)
        features = self.bottleneck_layer(features)
        outputs_adv, domain_output = None, None

        if self.use_adv:
            features_adv = self.grl_layer(features)
            domain_output = torch.sigmoid(self.domain_classifier(features_adv))
            outputs_adv = self.classifier_layer_2(features_adv)

        outputs = self.classifier_layer(features)
        output_prob = self.softmax(outputs)

        return features, outputs, output_prob, outputs_adv, domain_output

    def predict(self, inputs):
        features = self.base_network(inputs)
        features = self.bottleneck_layer(features)
        outputs = self.classifier_layer(features)
        output_prob = self.softmax(outputs)
        return output_prob, outputs


def mdd_loss(outputs, len_src, cls_adv, srcweight):
    class_criterion = torch.nn.CrossEntropyLoss()
    y_pred = outputs.max(1)[1]
    target_adv_src = y_pred[:len_src]
    target_adv_tgt = y_pred[len_src:]
    classifier_loss_adv_src = class_criterion(
        cls_adv[:len_src], target_adv_src)
    logloss_tgt = torch.log(torch.clamp(
        1 - F.softmax(cls_adv[len_src:], dim=1), min=1e-15))
    classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
    loss = srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
    return loss


class L2M(object):
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=1024, class_num=31, srcweight=3, use_adv=True, match_feat_type=5, dataset='Office-Home', cat_feature='column'):
        self.c_net = L2MNet(base_net, bottleneck_dim,
                            width, class_num, use_adv)
        self.is_train = False
        self.class_num = class_num
        self.srcweight = srcweight
        self.matched_feat = None
        self.use_adv = use_adv
        self.match_feat_type = match_feat_type
        self.dataset = dataset
        self.cat_feature = cat_feature
        self.label_src = None
        self.label_tar = None

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()
        len_src, len_tar = labels_source.size(
            0), inputs.size(0) - labels_source.size(0)
        feat, outputs, out_prob, outputs_adv, domain_output = self.c_net(
            inputs)
        classifier_loss = class_criterion(outputs[:len_src], labels_source)
        self.label_src = labels_source
        self.label_tar = out_prob[len_src:].max(1)[1]

        if self.use_adv:  # Adversarial-based
            cond_loss = mdd_loss(outputs, len_src, outputs_adv, self.srcweight)
            sdomain_label = torch.zeros(len_src).cuda().long()
            tdomain_label = torch.ones(len_tar).cuda().long()
            domain_label = torch.cat([sdomain_label, tdomain_label])
            mar_loss = class_criterion(domain_output, domain_label)
        else:  # MMD-based
            fea_src, fea_tar = feat[:len_src], feat[len_src:]
            conditional_loss = mmd.conditional(
                fea_src, fea_tar, labels_source, torch.nn.functional.softmax(outputs[len_src:], dim=1), classnum=self.class_num)
            marginal_loss = mmd.marginal(fea_src, fea_tar)
            mar_loss = marginal_loss
            cond_loss = conditional_loss

        return classifier_loss, cond_loss, mar_loss, feat, outputs

    def predict(self, inputs):
        _, _, softmax_outputs, _, _ = self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        if isinstance(self.c_net, torch.nn.DataParallel) or isinstance(self.c_net, torch.nn.parallel.DistributedDataParallel):
            return self.c_net.module.parameter_list
        return self.c_net.parameter_list

    def match_feat(self, cond_loss, mar_loss, feat, logits):
        if self.cat_feature == 'column':
            if self.match_feat_type == 0:  # feature
                self.matched_feat = feat
            elif self.match_feat_type == 1:  # logits
                self.matched_feat = logits
            elif self.match_feat_type == 2:  # conditional loss + marginal loss
                self.matched_feat = torch.cat(
                    [cond_loss.reshape([1, 1]), mar_loss.reshape([1, 1])], dim=1)
            elif self.match_feat_type == 3:  # feature + conditional loss + marginal loss
                loss = torch.cat([cond_loss.reshape([1, 1]).expand(
                    feat.size(0), 1), mar_loss.reshape([1, 1]).expand(feat.size(0), 1)], dim=1)
                self.matched_feat = torch.cat([feat, loss], dim=1)
            elif self.match_feat_type == 4:  # logits + conditional loss + marginal loss
                loss = torch.cat([cond_loss.reshape([1, 1]).expand(logits.size(
                    0), 1), mar_loss.reshape([1, 1]).expand(logits.size(0), 1)], dim=1)
                self.matched_feat = torch.cat([logits, loss], dim=1)
            elif self.match_feat_type == 5:  # feature + logits + conditional loss + marginal loss
                loss = torch.cat([cond_loss.reshape([1, 1]).expand(logits.size(
                    0), 1), mar_loss.reshape([1, 1]).expand(logits.size(0), 1)], dim=1)
                self.matched_feat = torch.cat([feat, logits, loss], dim=1)
            elif self.match_feat_type == 6:  # feature + label
                label = torch.cat(
                    [self.label_src.view(-1, 1).float(), self.label_tar.view(-1, 1).float()], dim=0)
                self.matched_feat = torch.cat([feat, label], dim=1)
        else:
            if self.match_feat_type == 0:  # feature
                self.matched_feat = feat
            elif self.match_feat_type == 1:  # logits
                self.matched_feat = logits
            elif self.match_feat_type == 2:  # conditional loss + marginal loss
                self.matched_feat = torch.cat(
                    [cond_loss.reshape([1, 1]), mar_loss.reshape([1, 1])], dim=1)
            elif self.match_feat_type == 3:  # feature + conditional loss + marginal loss
                loss = torch.cat([cond_loss.reshape([1, 1]),
                                  mar_loss.reshape([1, 1])], dim=1)
                wid = 512 if self.dataset == 'covid' else 2048
                loss = torch.cat([loss, torch.zeros([1, wid-2]).cuda()], dim=1)
                self.matched_feat = torch.cat([feat, loss], dim=0)
            elif self.match_feat_type == 4:  # logits + conditional loss + marginal loss
                d = logits.shape[1] - 2
                loss = torch.cat([cond_loss.reshape([1, 1]),
                                  mar_loss.reshape([1, 1])], dim=1)
                loss = torch.cat([loss, torch.zeros([1, d]).cuda()], dim=1)
                self.matched_feat = torch.cat([logits, loss], dim=0)
            elif self.match_feat_type == 5:  # feature + logits + conditional loss + marginal loss
                d = logits.shape[1] - 2
                loss = torch.cat([cond_loss.reshape([1, 1]),
                                  mar_loss.reshape([1, 1])], dim=1)
                loss = torch.cat([loss, torch.zeros([1, d]).cuda()], dim=1)
                self.matched_feat = torch.cat([feat, logits, loss], dim=0)
        return self.matched_feat
