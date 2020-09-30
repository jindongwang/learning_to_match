import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def split_line(line):
    items = line.strip().split(',')
    value = []
    for item in items:
        it = item.split(':').strip()
        value.append(float(it[-1]))
    return value

file_name = 'file.txt'
loss_mmd, loss_aug = [], []
with open(file_name, 'r') as fp:
    lines = fp.readlines()
    for item in lines:
        l = split_line(item)
        if item.__contains__('MMD'):
            loss_mmd.append([l[0], l[1]])
        else:
            loss_aug.append([l[0], l[1]])
print(np.array(loss_mmd))
print(np.array(loss_aug))

plt.plot(np.array(loss_mmd))