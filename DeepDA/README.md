# DeepDA: Deep Domain Adaptation Toolkit

A lightweight, easy-to-extend, easy-to-learn, high-performance, and for-fair-comparison toolkit based on PyTorch for domain adaptation (DA) of deep neural networks.


## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. Of course, you are welcome to add your algorithms here.

1. DAN (DDC) [1, 2]
2. DeepCoral [3]
3. DANN [4]
4. DSAN [5]
5. DAAN [6] (NEW!)

The detailed explanation of those methods are in the `README.md` files of corresponding directories.

## Installation

```
git clone https://github.com/jindongwang/transferlearning.git
cd code/DeepDA
pip install -r requirements.txt
```
We recommend to use `Python 3.7.10` which is our development environment.

## Usage

1. Modify the configuration file in the corresponding directories
2. Run the `main.py` with specified config, for example, `python main.py --config DAN/DAN.yaml`

* We provide shell scripts to help you reproduce our experimental results: `bash DAN/DAN.sh`.

## Customization

It is easy to design your own method following the 3 steps:

1. Check whether your method requires new loss functions, if so, add your loss in the `loss_funcs`
2. Check and write your own model's file inherited from our `models.TransferNet`
3. Write your own config file

## Results

We present results of our implementations on 2 popular benchmarks: Office-31 and Office-Home. We did not perform careful parameter tuning and simply used the default config files. You can easily reproduce our results using provided shell scripts!

1. Results on Office31

|     Method        | D - A | D - W | A - W | W - A | A - D  | W - D  | Average |
|-------------|-------|-------|-------|-------|--------|--------|---------|
| Source-only | 66.17 | 97.61 | 80.63 | 65.07 | 82.73  | 100.00 | 82.03   |
| DAN [1]         | 68.16 | 97.48 | 85.79 | 66.56 | 84.34  | 100.00 | 83.72   |
| DeepCoral [2]       | 66.06 | 97.36 | 80.25 | 65.32 | 82.53  | 100.00 | 81.92   |
| DANN [3]        | 67.06 | 97.86 | 84.65 | 71.03 | 82.73  | 100.00 | 83.89   |
| DSAN [4]        | 76.04 | 98.49 | 94.34 | 72.91 | 89.96  | 100.00 | 88.62   |


2. Results on Office-Home

|     Method       | A - C | A - P | A - R | C - A | C - P | C - R | P - A | P - C | P - R | R - A | R - C | R - P | Average |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| Source-only | 51.04 | 68.21 | 74.85 | 54.22 | 63.64 | 66.84 | 53.65 | 45.41 | 74.57 | 65.68 | 53.56 | 79.34 | 62.58   |
| DAN [1]        | 52.51 | 68.48 | 74.82 | 57.48 | 65.71 | 67.82 | 55.42 | 47.51 | 75.28 | 66.54 | 54.36 | 79.91 | 63.82   |
| DeepCoral [2]      | 52.26 | 67.72 | 74.91 | 56.20 | 64.70 | 67.48 | 55.79 | 47.17 | 74.89 | 66.13 | 54.34 | 79.05 | 63.39   |
| DANN [3]        | 51.48 | 67.27 | 74.18 | 53.23 | 65.10 | 65.41 | 53.15 | 50.22 | 75.05 | 65.35 | 57.48 | 79.45 | 63.12   |
| DSAN [4]        | 54.48 | 71.12 | 75.37 | 60.53 | 70.92 | 68.53 | 62.71 | 56.04 | 78.29 | 74.37 | 60.34 | 82.99 | 67.97   |


## Contribution
The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first.


## References

[1] Long, Mingsheng, et al. "Learning transferable features with deep adaptation networks." International conference on machine learning. PMLR, 2015.

[2] Tzeng, Eric, et al. "Deep domain confusion: Maximizing for domain invariance." arXiv preprint arXiv:1412.3474 (2014).

[3] Sun, Baochen, and Kate Saenko. "Deep coral: Correlation alignment for deep domain adaptation." European conference on computer vision. Springer, Cham, 2016.

[4] Ganin, Yaroslav, and Victor Lempitsky. "Unsupervised domain adaptation by backpropagation." International conference on machine learning. PMLR, 2015.

[5] Zhu, Yongchun, et al. "Deep subdomain adaptation network for image classification." IEEE transactions on neural networks and learning systems (2020).

[6] Yu, Chaohui, et al. "Transfer learning with dynamic adversarial adaptation network." 2019 IEEE International Conference on Data Mining (ICDM). IEEE, 2019.

## Citation
If you think this toolkit or the results are helpful to you and your research, please cite us!

```
@Misc{deepda,
howpublished = {\url{https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA}},   
title = {DeepDA: Deep Domain Adaptation Toolkit},  
author = {Wang, Jindong and Hou, Wenxin}
}  
```

## Contact

- [Wenxin Hou](https://houwenxin.github.io/): houwx001@gmail.com
- [Jindong Wang](http://www.jd92.wang/): jindongwang@outlook.com