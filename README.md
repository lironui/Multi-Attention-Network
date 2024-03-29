# Multiattention Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images

⭐ [Welcome to my HomePage](https://lironui.github.io/) ⭐ 

In this repository, a novel attention mechanism of kernel attention with linear complexity is proposed to alleviate the large computational demand in attention. Based on kernel attention and channel attention, we integrate local feature maps extracted by ResNet-50 with their corresponding global dependencies and reweight interdependent channel maps adaptively. Numerical experiments on two large-scale ﬁne-resolution remote sensing datasets demonstrate the superior performance of the proposed MANet.

The detailed results can be seen in the [Multi-Attention-Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/document/9487010).

The training and testing code can refer to [GeoSeg](https://github.com/WangLibo1995/GeoSeg).

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

| Method     | Backbone   | building | tree     | clutter   | road     | vegetation | static car | moving car | human    | mIoU     | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:| 
| MSD <sup>[1]</sup>      |- | 79.8     | 74.5     | 57.0      | 74.0     | 55.9       | 32.1       | 62.9       | 19.7 | 57.0     | 
| BANet <sup>[2]</sup>    |ResNet-18 | 85.4     | 78.9 | 66.6  | 80.7 | 62.1       | 52.8       | 69.3   | 21.0 | 64.6 | 
| MANet <sup>[3]</sup>    |ResNet-18 | 85.4     | 77.0     | 64.5      | 77.8     | 60.3       | 53.6       | 67.2        | 14.9      | 62.6    | 
| ABCNet <sup>[4]</sup>   |ResNet-18 | 86.4 | 79.9 | **67.4**  | **81.2** | 63.1       | 48.4       | 69.8   | 13.9     | 63.8 | 
| A<sup>2</sup>-FPN <sup>[5]</sup>|ResNet-18 | **87.2** | **80.1** | **67.4**  | 80.2 | 63.7       | 53.3       | **70.1**   | **23.4**     | **65.7** | 

Reference:  
`[1] Lyu, Y., Vosselman, G., Xia, G. S., Yilmaz, A., & Yang, M. Y. (2020). UAVid: A semantic segmentation dataset for UAV imagery. ISPRS journal of photogrammetry and remote sensing, 165, 108-119.`  
`[2] L. Wang, R. Li, D. Wang, C. Duan, T. Wang, and X. Meng. (2021). Transformer Meets Convolution: A Bilateral Awareness Network for Semantic Segmentation of Very Fine Resolution Urban Scene Images. Remote Sensing, 13(16), 3065, doi: 10.3390/rs13163065.`  
`[3] R. Li, S. Zheng, C. Zhang, C. Duan *, J. Su, L. Wang, and P. M. Atkinson. (2021). Multiattention network for semantic segmentation of fine-resolution remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3093977.`  
`[4] R. Li, S. Zheng, C. Zheng, C. Duan *, L. Wang, and P. M. Atkinson. ABCNet: Attentive bilateral contextual network for efficient semantic segmentation of Fine-Resolution remotely sensed imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 181, 84-98, doi: 10.1016/j.isprsjprs.2021.09.005.`  
`[5] R. Li, L. Wang, C. Zhang, C. Duan, S. Zheng. (2022) A2-FPN for semantic segmentation of fine-resolution remotely sensed images. International Journal of Remote Sensing, 43:3, 1131-1155, doi: 10.1080/01431161.2022.2030071.`


![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/vai.png)  
Fig. 2. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Vaihingen test set.

![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/pot.png)  
Fig. 3. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Potsdam test set.

Complexity:
------- 
![Complexity](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/consumer.png)  
Fig. 4. The (a) computation requirement and (b) memory requirement of the raw dot-product attention mechanism and the proposed kernel attention mechanism under different input sizes. Please notice that the figure is in log scale.
