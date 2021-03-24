import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mmd
from model.grl import GradReverse
from model.mlp import MLP
import model.backbone as backbone


class L2MNet(nn.Module):
    def __init__(self, base_net='ResNet50', bottleneck_dim=2048, width=256, class_num=31, use_adv=True):
        """Init func

        Args:
            base_net (str, optional): backbone network. Defaults to 'ResNet50'.
            bottleneck_dim (int, optional): bottleneck dim. Defaults to 2048.
            width (int, optional): width for fc layer. Defaults to 2048.
            class_num (int, optional): number of classes. Defaults to 31.
            use_adv (bool, optional): use adversarial training or not. Defaults to True.
        """
        super(L2MNet, self).__init__()
        self.use_adv = use_adv
        self.n_class = class_num
        self.featurizer = backbone.network_dict[base_net]()
        self.bottleneck_layer = nn.Linear(self.featurizer.output_num(), bottleneck_dim)
        self.classifier_layer = nn.Linear(bottleneck_dim, class_num)

        if self.use_adv:
            self.domain_classifier = MLP(
                bottleneck_dim, [bottleneck_dim, width], 2, drop_out=.5)
            
            self.domain_classifier_class = nn.ModuleList([MLP(
                bottleneck_dim, [bottleneck_dim, width], 2, drop_out=.5) for _ in range(class_num)])
        else:
            self.mmd = mmd.MMD_loss(class_num=self.n_class)

    def forward(self, inputs, label_src=None, compute_ada_loss=True):
        """Forward func

        Args:
            inputs (2d-array): raw inputs for both source and target domains
            label_src (1d-array, default: None): source domain labels
            compute_ada_loss (bool, default: False): compute adaptation loss or not
        Returns:
            outputs
        """
        features = self.featurizer(inputs)
        features = self.bottleneck_layer(features)
        out_logits = self.classifier_layer(features)
        out_prob = F.softmax(out_logits, dim=1)
        _, preds = torch.max(out_prob, 1)
        loss_cls = 0 if label_src is None else F.cross_entropy(out_logits[: label_src.size(0)], label_src)

        loss_mar, loss_cond = None, None
        if compute_ada_loss:
            if self.use_adv:
                feat_grl = GradReverse.apply(features, 1)
                # compute DANN loss
                loss_mar = self.dann(feat_grl)
                
                # compute class-wise DANN loss
                pred_src, pred_tar = preds[:label_src.size(0)], preds[label_src.size(0):]
                feat_src, feat_tar = feat_grl[: pred_src.size(0)], feat_grl[pred_src.size(0) : ]
                loss_cond, _ = self.adv_class(feat_src, feat_tar, pred_src, pred_tar)

            else:
                fea_src, fea_tar = features[:label_src.size(0)], features[label_src.size(0):]
                # compute mmd loss
                loss_mar = self.mmd.marginal_mmd(fea_src, fea_tar)

                # compute conditional mmd loss
                loss_cond = self.mmd.conditional_mmd(fea_src, fea_tar, label_src, out_prob[label_src.size(0):])
            
        return features, out_logits, preds, loss_cls, loss_mar, loss_cond

    def predict(self, inputs):
        """Prediction function

        Args:
            inputs (2d-array): raw input

        Returns:
            output_prob: probability
            outputs: logits
        """

        features = self.featurizer(inputs)
        features = self.bottleneck_layer(features)
        output_logit = self.classifier_layer(features)
        output_prob = F.softmax(output_logit)
        probs, preds = torch.max(output_prob, 1)
        return probs, output_logit, preds

    def dann(self, feat_grl):
        """Compute DANN loss

        Args:
            feat_grl (tensor): input features after GRL apply

        Returns:
            float: DANN loss
        """
        domain_output = torch.sigmoid(self.domain_classifier(feat_grl))
        len_src, len_tar = feat_grl.size(0) // 2, feat_grl.size(0) // 2
        sdomain_label = torch.zeros(len_src).cuda().long()
        tdomain_label = torch.ones(len_tar).cuda().long()
        domain_label = torch.cat([sdomain_label, tdomain_label])
        adv_loss = F.cross_entropy(domain_output, domain_label)
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
            loss_adv_i = F.cross_entropy(domain_out_i, domain_label)
            losses.append(loss_adv_i)
        loss_avg = torch.tensor(losses, device='cuda').mean()
        return loss_avg, losses

if __name__ == '__main__':
    l2m_model = L2M()
    inputs = torch.randn(4,3,224,224)
    print(l2m_model.predict(inputs))

