# Multi-Attention-Network

In this repository, a novel attention mechanism of kernel attention with linear complexity is proposed to alleviate the large computational demand in attention. Based on kernel attention and channel attention, we integrate local feature maps extracted by ResNet-50 with their corresponding global dependencies and reweight interdependent channel maps adaptively. Numerical experiments on two large-scale ﬁne-resolution remote sensing datasets demonstrate the superior performance of the proposed MANet.

The detailed results can be seen in the [Multi-Attention-Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images](https://ieeexplore.ieee.org/document/9487010).

The related repositories include:
* [MACU-Net](https://github.com/lironui/MACU-Net)->A revised U-Net structure.
* [MAResU-Net](https://github.com/lironui/MAResU-Net)->Another type of attention mechanism with linear complexity.

If our code is helpful to you, please cite:

`Li, R., Zheng, S., Duan, C., Zhang, C., Su, J., & Atkinson, P. M. (2021), "Multiattention Network for Semantic Segmentation of Fine-Resolution Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3093977.`

Our paper has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS). The link about the citation will be updated as soon as possible. 


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
![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/vai.png)  
Fig. 2. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Vaihingen test set.

![Vaihingen](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/pot.png)  
Fig. 3. Qualitative comparisons (1024 × 1024 patches) between our method and baseline on Potsdam test set.

Complexity:
------- 
![Complexity](https://github.com/lironui/Multi-Attention-Network/blob/master/Fig/consumer.png)  
Fig. 4. The (a) computation requirement and (b) memory requirement of the raw dot-product attention mechanism and the proposed kernel attention mechanism under different input sizes. Please notice that the figure is in log scale.
