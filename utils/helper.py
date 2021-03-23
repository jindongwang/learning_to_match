import numpy as np
import random
import torch

# controls random seed
def set_random(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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