# SEGAN
![](/pictures/figure4.png)

From left to right: baseline, comparison method 1, this project's method, comparison method 2
***
# Contents
- [Overview](#Overview)
  - [LatentSpace](#LatentSpace)
  - [LinearSubSpace](#LinearSubSpace)
- [Run](#Run)
- [Model](#Model)
- [Dataset](#Dataset)
- [Discussion](#Discussion)
  - [Baseline&OtherDesign](#Baseline&OtherDesign)
  - [Result](#Result)
***
## Overview
`SEGAN` is a project aims to control semantic attributes of results generated by [StyleGAN2](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html) through modified it's latent space. In their paper, they discussed the impact of latent space on the results of human face images, lower layers affect more general semantic attributes, such as gender, skin color, and higher layers affect in details, such as smiles, hairstyles, etc. It is worth noting that the latent space here is not equal to the space of the initial noise sampled in the original [GAN](https://proceedings.neurips.cc/paper_files/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html), we will discuss it more in next section. They used this discovery to do "StyleMix", it can mix styles from different seed images to cantrol the semantic attributes of new generated images. But this control is not precise, it is difficult to decouple semantic attributes and control them independently in this way. Therefore, SEGAN introduces a linear subspace to locate interpretable and controllable dimensions from vectors from latent space.
### LatentSpace
[StyleGAN](https://openaccess.thecvf.com/content_CVPR_2019/html/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.html) has a intermediate latent space $W$, which then controls the generator through adaptive instance normalization (AdaIN) at each convolution layer in their unique network architexture.

![](/pictures/figure10.png)
![](/pictures/figure9.png)
### LinearSubSpace
The idea of ​​using linear subspace comes from [EigenGAN](https://openaccess.thecvf.com/content/ICCV2021/html/He_EigenGAN_Layer-Wise_Eigen-Learning_for_GANs_ICCV_2021_paper.html), showed that linear subspace can find interpretable and controllable dimensions from different generator layers. They succeeded on the original GAN, so the questions here is how to apply this method in StyleGAN2 which have a quite different structure. The network architecture will be discussed in later section.
***
## Run
Refer to this [note](https://colab.research.google.com/drive/1Zil4-l8Lvz9cYKpsVG_0IMatErMcUi_L?usp=sharing).
#### Hyperparameters
- resolution: 64
- learning rate: 0.002
- batch size: 16
- dimension of latent vecotr: 64
- r1 weight: 10
- regularzation interval of discriminator: 16
- regularzation interval of generator: 4
***
## Model
Directly adding a linear subspace in each block and add the output to the feature map as in the original GAN ​​does not fully apply to the case of StyleGAN2. This would make the network too complex, and more importantly we want to modify semantic attributes by modifying $W$ space instead of $Z$ space (sampling noise). The following figure shows model structure whitch have a best performance. The linear subspace is placed before the modulation module of each layer and processes $w$ vectors. Use `--mode=2` to switch to this model.

![](/pictures/figure2.png)
***
## Dataset
The dataset can be found [here](https://drive.google.com/file/d/1uun17wO53E0gRUouAizs7kV4bUOO-6yy/view?usp=sharing), It's a very small subset of the danbooru dataset.
***
## Discussion
### Baseline&OtherDesign
The baseline is original StyleGAN2, use `--mode=0` to switch to it. There are still two modes as comparison, `--mode=1` and --mode=3, their model structure showed in following figures.

![](/pictures/figure1.png)
![](/pictures/figure3.png)
### Result
#### Generate Images
Some samples generated by mode 0 to 3, left to right.

![](/pictures/figure4.png)
#### Training
![](/pictures/figure7.png)
![](/pictures/figure8.png)

*The red curve's (mode=2) log have some trouble, need to retrain it. TODO*
|Model|FID|
|----|----|
|origin(0)|133.75|
|1|139.33|
|this project's|123.79|
|2|163.29|
#### Decoupling & Semantic Control
$PPL=E[1/ϵ^2  d(G(slerp(z1,z2;t)),G(slerp(z1,z2;t+ϵ)))$
|Model|PPL($ϵ=1e-2,t∈(0,1),30ktimes$)|
|----|----|
|origin(0)|821.70|
|1|816.25|
|this project's|818.62|
|2|824.03|

![](/pictures/figure6.png)
Above figure shows control semantic attributes of generated results by modified their latent vector in specific layer and dimensions which found by linear space. Where L means layer, N means number of linear subspace, D means dimension.
***
## Others
Here's no regularzation in linear space so actually the implementation of linear space in this project is different with EigenGAN, have to fix it fulture.TODO
