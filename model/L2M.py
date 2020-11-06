import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
from utils import mmd
from model.grl import GradReverse
from model.mlp import MLP

'''
L2M network.
'''
class L2M(nn.Module):
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=2048, class_num=31, use_adv=True, match_feat_type=0):
        """Init func

        Args:
            base_net (str, optional): backbone network. Defaults to 'ResNet50'.
            bottleneck_dim (int, optional): bottleneck dim. Defaults to 2048.
            width (int, optional): width for fc layer. Defaults to 2048.
            class_num (int, optional): number of classes. Defaults to 31.
            use_adv (bool, optional): use adversarial training or not. Defaults to True.
        """
        super(L2M, self).__init__()
        self.use_adv = use_adv
        self.n_class = class_num
        self.match_feat_type = match_feat_type
        # set base network
        self.base_network = backbone.network_dict[base_net]()
        self.bottleneck_layer = nn.Sequential(*[nn.Linear(self.base_network.output_num(), bottleneck_dim),
                                                nn.BatchNorm1d(bottleneck_dim),
                                                nn.ReLU(),
                                                nn.Dropout(0.5, inplace=False)])
        self.classifier_layer = nn.Sequential(*[nn.Linear(bottleneck_dim, width),
                                                nn.ReLU(),
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
                               {"params": self.bottleneck_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1}]

        if self.use_adv:
            self.classifier_layer_2 = MLP(
                bottleneck_dim, [width], class_num, drop_out=.5)
            # DANN: add DANN, make L2M not only class-invariant (as conditional loss) but also domain invariant (as marginal loss)
            self.domain_classifier = MLP(
                bottleneck_dim, [bottleneck_dim, width], 2, drop_out=.5)
            
            # Class-conditional DANN
            self.domain_classifier_class = nn.ModuleList([MLP(
                bottleneck_dim, [bottleneck_dim, width], 2, drop_out=.5) for _ in range(class_num)])

            self.parameter_list.append(
                {"params": self.classifier_layer_2.parameters(), "lr": 1})
            self.parameter_list.append(
                {"params": self.domain_classifier.parameters(), "lr": 1})
            self.parameter_list.append(
                {"params": self.domain_classifier_class.parameters(), "lr": 1})

    def forward(self, inputs, label_src):
        """Forward func

        Args:
            inputs (2d-array): raw inputs for both source and target domains

        Returns:
            outputs
        """
        class_criterion = nn.CrossEntropyLoss()
        f = self.base_network(inputs)
        features = self.bottleneck_layer(f)
        out_logits = self.classifier_layer(features)
        out_prob = self.softmax(out_logits)
        _, preds = torch.max(out_prob, 1)
        loss_cls = class_criterion(out_logits[: label_src.size(0)], label_src)

        outputs_adv = None
        loss_mar, loss_cond = None, None
        if self.use_adv:
            feat_grl = GradReverse.apply(features, 1)
            # compute DANN loss
            loss_mar = self.dann(feat_grl)
            
            # compute class-wise DANN loss
            pred_src, pred_tar = preds[:label_src.size(0)], preds[label_src.size(0):]
            feat_src, feat_tar = feat_grl[: pred_src.size(0)], feat_grl[pred_src.size(0) : ]
            loss_cond, _ = self.adv_class(feat_src, feat_tar, pred_src, pred_tar)

            # compute inputs for mdd
            outputs_adv = self.classifier_layer_2(feat_grl)
        else:
            fea_src, fea_tar = features[:label_src.size(0)], features[label_src.size(0):]
            # compute mmd loss
            loss_mar = mmd.marginal(fea_src, fea_tar)

            # compute conditional mmd loss
            loss_cond = mmd.conditional(fea_src, fea_tar, label_src, out_prob[label_src.size(0):], classnum=self.n_class)
            
        return features, out_logits, out_prob, outputs_adv, preds, loss_cls, loss_mar, loss_cond

    def dann(self, feat_grl):
        """Compute DANN loss

        Args:
            feat_grl (tensor): input features after GRL apply

        Returns:
            float: DANN loss
        """
        class_criterion = nn.CrossEntropyLoss()
        domain_output = torch.sigmoid(self.domain_classifier(feat_grl))
        len_src, len_tar = feat_grl.size(0) // 2, feat_grl.size(0) // 2
        sdomain_label = torch.zeros(len_src).cuda().long()
        tdomain_label = torch.ones(len_tar).cuda().long()
        domain_label = torch.cat([sdomain_label, tdomain_label])
        adv_loss = class_criterion(domain_output, domain_label)
        return adv_loss

    def adv_class(self, feat_s, feat_t, label_s, label_t):
        """Compute class-wise DANN loss

        Args:
            feat_s (matrix): input source features after GRL apply
            feat_t (matrix): input target features after GRL apply
            label_s (1d array): source labels
            label_t (1d array): target labels

        Returns:
            tuple: Loss and loss list for each class
        """
        class_criterion = nn.CrossEntropyLoss()
        losses = []
        for label in torch.unique(label_s):
            feat_src, feat_tar = feat_s[label_s == label], feat_t[label_t == label]
            if feat_src.size(0) == 0 or feat_tar.size(0) == 0:
                continue
            feat_i = torch.cat([feat_src, feat_tar], dim=0)
            domain_out_i = torch.sigmoid(self.domain_classifier_class[label](feat_i))
            sdomain_label = torch.zeros(feat_src.size(0)).cuda().long()
            tdomain_label = torch.ones(feat_tar.size(0)).cuda().long()
            domain_label = torch.cat([sdomain_label, tdomain_label])
            loss_adv_i = class_criterion(domain_out_i, domain_label)
            losses.append(loss_adv_i)
        loss_avg = sum(losses) / len(torch.unique(label_s))
        return loss_avg, losses

    def predict(self, inputs):
        """Prediction function

        Args:
            inputs (2d-array): raw input

        Returns:
            output_prob: probability
            outputs: logits
        """
        f = self.base_network(inputs)
        features = self.bottleneck_layer(f)
        output_logit = self.classifier_layer(features)
        output_prob = self.softmax(output_logit)
        probs, preds = torch.max(output_prob, 1)
        return probs, output_logit, preds

    def get_parameter_list(self):
        if isinstance(self, torch.nn.DataParallel) or isinstance(self, torch.nn.parallel.DistributedDataParallel):
            return self.module.parameter_list
        return self.parameter_list

    def match_feat(self, cond_loss, mar_loss, feat, logits):
        """Generate matching features

        Args:
            cond_loss (float): conditional loss
            mar_loss (float): marginal loss
            feat (2d-array): feature matrix
            logits (1d-aray): logits

        Returns:
            matching features
        """
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
        
        return self.matched_feat

def mdd_loss(outputs, len_src, cls_adv, srcweight):
    """Implement the MDD loss. Reference:
    Zhang et al. Bridging theory and algorithm for domain adpatation. ICML 2019.

    Args:
        outputs (1d-array): logits
        len_src (int): length of source domain data
        cls_adv (1d-array): adversarial logits of two domains
        srcweight (float): source domain weight for mdd loss

    Returns:
        loss (float): mdd loss
    """
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

