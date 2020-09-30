import torch
import numpy as np
from IPython import embed
min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    #loss = 0.0
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    '''loss = torch.mean(torch.mm(f_of_X, torch.transpose(f_of_X, 0, 1)) +
                      torch.mm(f_of_Y, torch.transpose(f_of_Y, 0, 1)) -
                      2 * torch.mm(f_of_X, torch.transpose(f_of_Y, 0, 1)))'''
    # loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    #delta = f_of_X - f_of_Y
    #loss = torch.mean((delta * delta).sum(1))
    # print(loss)
    return loss


def c_linear_mmd2(f_of_X, f_of_Y, s_label, t_label):
    s_label = s_label.cpu()
    batch_size = f_of_X.size()[0]
    label_size = t_label.size()[1]
    s_label = s_label.view(batch_size, 1)
    s_label = torch.zeros(batch_size, label_size).scatter_(1, s_label.data, 1)
    s_label = s_label.cuda()
    t_label = t_label.data.cuda()
    a = 1
    b = 0
    loss = (a * torch.mm(s_label, torch.transpose(s_label, 0, 1)) +
            b) * torch.mm(f_of_X, torch.transpose(f_of_X, 0, 1))
    loss += (a * torch.mm(t_label, torch.transpose(t_label, 0, 1)) +
             b) * torch.mm(f_of_Y, torch.transpose(f_of_Y, 0, 1))
    loss -= (a * torch.mm(s_label, torch.transpose(t_label, 0, 1)) +
             b) * 2 * torch.mm(f_of_X, torch.transpose(f_of_Y, 0, 1))
    loss = torch.mean(loss)
    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def marginal(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def convert_to_onehot(sca_label, class_num=65):
    return np.eye(class_num)[sca_label]

class Weight:

    @staticmethod
    def cal_weight(s_label, t_label, batch_size=32, class_num=65):
        batch_size = s_label.size()[0]
        s_sca_label = s_label.cpu().data.numpy()
        s_vec_label = convert_to_onehot(s_sca_label, class_num)
        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        s_sum[s_sum == 0] = 100
        s_vec_label = s_vec_label / s_sum

        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        #t_vec_label = convert_to_onehot(t_sca_label)

        t_vec_label = t_label.cpu().data.numpy()
        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        t_sum[t_sum == 0] = 100
        t_vec_label = t_vec_label / t_sum

        weight_ss = np.zeros((batch_size, batch_size))
        weight_tt = np.zeros((batch_size, batch_size))
        weight_st = np.zeros((batch_size, batch_size))

        set_s = set(s_sca_label)
        set_t = set(t_sca_label)
        count = 0
        for i in range(class_num):
            if i in set_s and i in set_t:
                s_tvec = s_vec_label[:, i].reshape(batch_size, -1)
                t_tvec = t_vec_label[:, i].reshape(batch_size, -1)
                ss = np.dot(s_tvec, s_tvec.T)
                weight_ss = weight_ss + ss# / np.sum(s_tvec) / np.sum(s_tvec)
                tt = np.dot(t_tvec, t_tvec.T)
                weight_tt = weight_tt + tt# / np.sum(t_tvec) / np.sum(t_tvec)
                st = np.dot(s_tvec, t_tvec.T)
                weight_st = weight_st + st# / np.sum(s_tvec) / np.sum(t_tvec)
                count += 1

        length = count  # len( set_s ) * len( set_t )
        if length != 0:
            weight_ss = weight_ss / length
            weight_tt = weight_tt / length
            weight_st = weight_st / length
        else:
            weight_ss = np.array([0])
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return torch.from_numpy(weight_ss.astype('float32')).cuda(), torch.from_numpy(weight_tt.astype('float32')).cuda(), torch.from_numpy(weight_st.astype('float32')).cuda()


def conditional(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, classnum=-1):
    batch_size = source.size()[0]
    weight_ss, weight_tt, weight_st = Weight.cal_weight(
        s_label, t_label, class_num=classnum)

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = torch.Tensor([0]).cuda()
    if torch.sum(torch.isnan(sum(kernels))):
        return loss
    SS = kernels[:batch_size, :batch_size]
    TT = kernels[batch_size:, batch_size:]
    ST = kernels[:batch_size, batch_size:]

    loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    return loss
