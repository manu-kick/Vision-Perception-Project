# Vision-Perception-Project
This is the repo for the vision and perception project
A Pytorch Lightning reimplementation of CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding (CVPR'22)
Link to the paper -> https://arxiv.org/abs/2203.00680

## Dependencies
Refer to the requirements.txt file


## How to Use
Note: In the following guide we are supposing you would run the code on colab as we did, then it's likely that you will have errors regarding path;
To avoid errors be sure to have this exactly path as soon as you give access Colab to Drive: "content/drive/MyDrive/.../"

Here you find all the first step to download configure and run the code to train the models or to use our pretrained model

- clone repository from GitHub: git clone https://github.com/manu-kick/Vision-Perception-Project.git
- download the datasets used in our experiment from this link https://drive.google.com/drive/folders/15aYNCSQAiCNuluOJEUUMP3SBalsqrUCL?usp=sharing
- insert the all the folders downloaded in the 'data' folder in the root of the repo have(you should have the following folders: modelnet40_ply_hdf5_2048, ScanObjectNN, ShapeNet, ShapeNetRendering), make sure to unzip the files
- then you can upload everything on drive, you must call the root repo folder with the following name: "Vision-Perception-Project"


## Train the model you like
You can open the V2_Crosspoint_Lightning.ipynb to train the models, you will find details instrunction inside that file


## Pretrained model
Here you find our pretrained model, in every folder you find the point model needed for the evalutation (you will find also the image feature extractor)
https://drive.google.com/drive/folders/10Ay2y9zo5fj6n2KfqRlWkv3IPqm3pqKR?usp=sharing

## Evaluate the models
You can use the eval_ssl file to evaluate the model, you find the instruction

## Our report
You find in eval_ssl some observations about the results we get

## Acknowledgements % references
Our code borrows heavily from:
[DGCNN](https://github.com/WangYueFt/dgcnn) 
[PCT]https://github.com/qinglew/PointCloudTransformer
REFERENCES
[1] CrossPoint: Self-Supervised Cross-Modal Contrastive Learning for 3D Point Cloud Understanding
Mohamed Afham, Isuru Dissanayake, Dinithi Dissanayake, Amaya Dharmasiri, Kanchana Thilakarathna, Ranga Rodrigo
[2] PCT: Point cloud transformer
Meng-Hao Guo, Jun-Xiong Cai, Zheng-Ning Liu, Tai-Jiang Mu, Ralph R. Martin, Shi-Min Hu
[3] Dynamic Graph CNN for Learning on Point Clouds
Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon
[4] Vision Transformers: State of the Art and Research Challenges, Bo-Kai Ruan, Hong-Han Shuai, Wen-Huang Cheng		
[5] Deep Residual Learning for Image Recognition
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
