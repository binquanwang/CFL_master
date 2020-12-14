
This repo contains the source code for our work
"Central Feature Learning for Unsupervised Person Re-identification"




### Prerequisites
1. Pytorch 0.4.1
2. Python 3.6



### Preparation
 
- Data preparation

```bash {.line-numbers}
mkdir data

ln -s [PATH TO MSMT17_V1] ./data/MSMT17_V1
ln -s [PATH TO DUKE] ./data/DukeMTMC-reID
ln -s [PATH TO Market] ./data/Market
```

- set the path of ImageNet pretrained models
```bash {.line-numbers}
ln -s [THE PATH OF IMAGENET PRE-TRAINED MODELS] imagenet_models
```
### Run the code
- For pretraining the model
```bash {.line-numbers}
cd ./train
python supervised_train.py --gpu [CHOOSE WHICH GPU TO RUN] --exp-name [YOUR EXP NAME]
```


```bash
mkdir ./snapshot
mkdir ./snapshot/MSMT17_PRE
cp [PATH TO PRETRAINED MODEL] ./snapshot/MSMT17_PRE/
# it means the name of the experiment of pretraining is 'MSMT17_PRE'  
```

- For unsupervised training
```bash {.line-numbers}
cd ./unsupervised

# for market
python unsupervised_train.py --data MARKET --gpu [CHOOSE WHICH GPU TO RUN] \
--pre-name [THE EXP NAME OF PRE-TRIANED MODEL] --exp-name [YOUR EXP NAME] \
--batch-size 42 --scale 15 --lr 0.0001 

 # for duke
python unsupervised_train.py --data DUKE --gpu [CHOOSE WHICH GPU TO RUN] \
--pre-name [THE EXP NAME OF PRE-TRIANED MODEL] --exp-name [YOUR EXP NAME] \
--batch-size 40 --scale 5 --lr 0.0001 

```

### Code link

the link to the specific code of each comparison method [ECN](https://github.com/zhunzhong07/ECN) \
 [PAUL](https://github.com/QizeYang/PAUL) \
 [HHL](https://github.com/zhunzhong07/HHL) \
 [MAR](https://github.com/KovenYu/MAR) \
 [BUC](https://github.com/L1aoXingyu/Bottom-up-Clustering-Person-Re-identification) \
 [PCB-PAST](https://github.com/zhangxinyu-xyz/PAST-ReID) \
 



### Reference

If you find our work helpful in your research,
please kindly cite our paper:

Qize Yang, Hong-Xing Yu, Ancong Wu, Wei-Shi Zheng, "Patch-based discriminative feature 
learning for unsupervised person re-identification",
In CVPR, 2019.

Zhun Zhong, Liang Zheng, Shaozi Li and Yi Yang, "Generalizing a person retrieval model hetero-and homogeneously",
ECCV, 2018.

Zhun Zhong and Liang Zheng and Zhiming Luo and Shaozi Li1 and Yi Yang, "Invariance matters: exemplar memory for domain adaptive person re-identification",
CVPR, 2019.






