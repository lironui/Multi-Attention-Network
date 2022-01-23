# Multi-Attention-Network

⭐ [Welcome to my HomePage](https://lironui.github.io/) ⭐ 

In this repository, a novel attention mechanism of kernel attention with linear complexity is proposed to alleviate the large computational demand in attention. Based on kernel attention and channel attention, we integrate local feature maps extracted by ResNet-50 with their corresponding global dependencies and reweight interdependent channel maps adaptively. Numerical experiments on two large-scale ﬁne-resolution remote sensing datasets demonstrate the superior performance of the proposed MANet.

The detailed results can be seen in the [Multi-Attention-Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/document/9487010).

The related repositories include:
* [MACU-Net](https://github.com/lironui/MACU-Net)->A revised U-Net structure.
* [MAResU-Net](https://github.com/lironui/MAResU-Net)->Another type of attention mechanism with linear complexity.

If our code is helpful to you, please cite:

`Li, R., Zheng, S., Duan, C., Zhang, C., Su, J., & Atkinson, P. M. (2021), "Multiattention Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3093977.`

Acknowlegement:
------- 
Thanks very much for the sincere help from Jianlin Su as well as his blog [线性Attention的探索：Attention必须有个Softmax吗？](https://spaces.ac.cn/archives/7546)


Requirements：
------- 
```
numpy >= 1.16.5
PyTorch >= 1.3.1
sklearn >= 0.20.4
tqdm >= 4.46.1
imageio >= 2.8.0
```

Network:
------- 
![network](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/MultiNet.png)  
Fig. 1.  The structure of (a) the proposed MANet, (b) the Attention block, (c) the ResBlock, and (d) the DeBlock.

Result:
------- 
The result on the [UAVid dataset](https://uavid.nl/) can seen from [here](https://competitions.codalab.org/competitions/public_submissions/25224) or download by this [link](https://competitions.codalab.org/my/competition/submission/975606/input.zip):

| Method    | Backbone  | building | tree     | clutter   | road     | vegetation | static car | moving car | human    | mIoU     | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| 
| MSD       |- | 79.8     | 74.5     | 57.0      | 74.0     | 55.9       | 32.1       | 62.9       | 19.7 | 57.0     | 
| BiSeNet   |ResNet-18 | 85.7     | 78.3     | 64.7      | 61.1     | **77.3**   | **63.4**   | 48.6       | 17.5     | 61.5     | 
| SwiftNet  |ResNet-18 | 85.3     | 78.2     | 64.1      | 61.5     | 76.4       | 62.1       | 51.1       | 15.7     | 61.1     | 
| ShelfNet  |ResNet-18 | 85.3     | 78.2     | 44.1      | 61.4     | 43.4       | 21.0       | 52.6       | 3.6      | 47.0     | 
| MANet     |ResNet-18 | 85.4     | 77.0     | 64.5      | 77.8     | 60.3       | 53.6       | 67.2        | 14.9      | 62.6    | 
| BANet     |ResNet-18 | 85.4     | 78.9 | 66.6  | 80.7 | 62.1       | 52.8       | 69.3   | 21.0 | 64.6 | 
| ABCNet    |ResNet-18 | 86.4 | 79.9 | **67.4**  | **81.2** | 63.1       | 48.4       | 69.8   | 13.9     | 63.8 | 
| A<sup>2</sup>-FPN |ResNet-18 | **87.2** | **80.1** | **67.4**  | 80.2 | 63.7       | 53.3       | **70.1**   | **23.4**     | **65.7** | 

![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/vai.png)  
Fig. 2. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Vaihingen test set.

![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/pot.png)  
Fig. 3. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Potsdam test set.

Complexity:
------- 
![Complexity](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/consumer.png)  
Fig. 4. The (a) computation requirement and (b) memory requirement of the raw dot-product attention mechanism and the proposed kernel attention mechanism under different input sizes. Please notice that the figure is in log scale.
