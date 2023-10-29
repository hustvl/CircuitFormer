<div align="center">
<h1>CircuitFormer </h1>
<h3>Circuit as Set of Points</h3>

Jialv Zou<sup>1</sup> , Xinggang Wang<sup>1 :email:</sup> , Jiahao Guo<sup>1</sup> , Wenyu Liu<sup>1</sup> , Qian Zhang<sup>2</sup> , Chang Huang<sup>2</sup>
 
<sup>1</sup> School of EIC, HUST, <sup>2</sup> Horizon Robotics

(<sup>:email:</sup>) corresponding author.

**NeurIPS 2023**

[ArXiv Preprint](https://arxiv.org/abs/2310.17418)

</div>

# Abstract
As the size of circuit designs continues to grow rapidly, artificial intelligence technologies are being extensively used in Electronic Design Automation (EDA) to assist with circuit design.
Placement and routing are the most time-consuming parts of the physical design process, and how to quickly evaluate the placement has become a hot research topic. 
Prior works either transformed circuit designs into images using hand-crafted methods and then used Convolutional Neural Networks (CNN) to extract features, which are limited by the quality of the hand-crafted methods and could not achieve end-to-end training, or treated the circuit design as a graph structure and used Graph Neural Networks (GNN) to extract features, which require time-consuming preprocessing.
In our work, we propose a novel perspective for circuit design by treating circuit components as point clouds and using Transformer-based point cloud perception methods to extract features from the circuit. This approach enables direct feature extraction from raw data without any preprocessing, allows for end-to-end training, and results in high performance.
Experimental results show that our method achieves state-of-the-art performance in congestion prediction tasks on both the CircuitNet and ISPD2015 datasets, as well as in design rule check (DRC) violation prediction tasks on the CircuitNet dataset.
Our method establishes a bridge between the relatively mature point cloud perception methods and the fast-developing EDA algorithms, enabling us to leverage more collective intelligence to solve this task.

# Installation

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n circuitformer python=3.9 -y
conda activate circuitformer
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install torch_scatter and spconv.**
  * You could install torch_scatter with pip, see the official documents of [torch_scatter](https://github.com/rusty1s/pytorch_scatter).
  * You could install latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).

**d. Install other requirements.**
```shell
pip install -r requirement.txt
```

**e. Prepare pretrained models.**
```shell
mkdir ckpts

cd ckpts 
wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```
# Prepare Dataset
Please download CircuitNet-N28 dataset follow the guide of [CircuitNet](https://github.com/circuitnet/CircuitNet)

**Folder structure**
```
circuitformer
├── dataset/
│   ├── CircuitNet/
│   │   ├── graph_features/
|   |   |   ├── instance_placement/
│   │   ├── train_congestion/
│   │   │   ├── congestion/
│   │   │   |   ├── feature/
│   │   │   |   ├── label/
│   │   ├── train_DRC/
│   │   │   ├── DRC/
│   │   │   |   ├── feature/
│   │   │   |   ├── label/
```

# Train and Test
Train circuitformer
```
python train.py
```
Please download our pretrain model [Here](https://pan.baidu.com/s/106j2W5VF2ehzaXpLhVoQpA?pwd=c16q) and put it in 'ckpts/'

Test circuitformer
```
python test.py
```

# Acknowledgment
The dataset is provided by [CircuitNet](https://github.com/circuitnet/CircuitNet). Our code is developed based on [VoxSeT](https://github.com/skyhehe123/VoxSeT) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
