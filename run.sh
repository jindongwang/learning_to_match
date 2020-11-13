#!/bin/bash  
  
for lamb in 0.01 0.05 0.1 0.5 1 5 10;  
do
    for mu in 0.5 1;
    do
        echo $lamb-$mu
        CUDA_VISIBLE_DEVICES=3 python train_new.py --dataset viral --root_path /home/jindwang/mine/data/viral --batch_size 16 --meta_m 8 --gbatch 16 --exp viral-ours --seed 52 --use_adv True --lamb $lamb --mu $mu
    done
done