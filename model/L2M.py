import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import mmd
from model.grl import Discriminator, AdversarialLoss
import model.backbone as backbone

'''
L2M network.
'''
class L2M(nn.Module):
    def __init__(self, base_net='resnet50', bottleneck_dim=2048, width=256, class_num=31, use_adv=True):
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
        self.featurizer = backbone.get_backbone(base_net)
        self.bottleneck_layer = nn.Sequential(*[nn.Linear(self.featurizer.output_num(), bottleneck_dim),
                                                nn.ReLU()])
        self.classifier_layer = nn.Linear(bottleneck_dim, self.n_class)

        self.parameter_list = self.get_parameters()
        if self.use_adv:
            # DANN: add DANN, make L2M not only class-invariant (as conditional loss) but also domain invariant (as marginal loss)
            self.domain_classifier = Discriminator(input_dim=bottleneck_dim)
            

            self.parameter_list.append(
                {'params': self.domain_classifier.parameters(), 'lr': 1.0 * 1}
            )
            
            self.adv_loss = AdversarialLoss(domain_classifier=self.domain_classifier)
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
                # compute DANN loss
                
                loss_mar = self.adv_loss(features[:label_src.size(0)], features[label_src.size(0):])
                loss_cond = 0

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

    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.featurizer.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        # if self.transfer_loss == "adv":
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        # elif self.transfer_loss == "daan":
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        #     params.append(
        #         {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
        #     )
        return params

if __name__ == '__main__':
    l2m_model = L2M()
    inputs = torch.randn(4,3,224,224)
    print(l2m_model.predict(inputs))

