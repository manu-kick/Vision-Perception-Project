import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import random
from PIL import Image
from .plyfile import load_ply
from . import data_utils as d_utils
import torchvision.transforms as transforms
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_DIR = 'data/ShapeNetRendering'

trans_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])
    
trans_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
                d_utils.PointcloudRotate(),
                d_utils.PointcloudTranslate(0.5, p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
            ])

trans_val = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(lo=0.5, hi=2, p=1),
            ])

def load_modelnet_data(partition):
    BASE_DIR = './drive/MyDrive/VP_CrossPoint_2023'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def load_ScanObjectNN(partition):
    #BASE_DIR = 'data/ScanObjectNN'
    BASE_DIR = './drive/MyDrive/VP_CrossPoint_2023/data/ScanObjectNN'
    DATA_DIR = os.path.join(BASE_DIR, 'main_split')
    h5_name = os.path.join(DATA_DIR, f'{partition}.h5')
    f = h5py.File(h5_name)
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    
    return data, label

def load_shapenet_data(exclusion_list=[]):
    BASE_DIR = './drive/MyDrive/VP_CrossPoint_2023'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    print("Listdir base dir: ", os.listdir(BASE_DIR))
    print(f'Data dir: {DATA_DIR}')
    all_filepath = []
    classes = glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*'))
    print(f"classes dir: {classes}")
    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet/*')):
        print(f'Class: {cls}')
        elements_list = cls.split(os.sep)
        current_class = elements_list[len(elements_list)-1]
        pcs = glob.glob(os.path.join(cls, '*'))
        if current_class not in exclusion_list:
            all_filepath += pcs
        
    return all_filepath

def get_render_imgs(pcd_path):
    path_lst = pcd_path.split(os.sep)
    path_lst[-1] = path_lst[-1][:-4]
    path_lst.append('rendering')
    path_lst[5] = 'ShapeNetRendering'

    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))
    #print(DIR)
    return img_path_list
        
class ShapeNetRender(Dataset):
    def __init__(self, img_transform = None, n_imgs = 1):
        self.data = load_shapenet_data()
        self.transform = img_transform
        self.n_imgs = n_imgs

    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        render_img_path = random.choice(get_render_imgs(pcd_path))
        # render_img_path_list = random.sample(get_render_imgs(pcd_path), self.n_imgs)
        # render_img_list = []
        # for render_img_path in render_img_path_list:
        render_img = Image.open(render_img_path).convert('RGB')
        render_img = self.transform(render_img)  #.permute(1, 2, 0)
            # render_img_list.append(render_img)
        pointcloud_1 = load_ply(self.data[item])
        # pointcloud_orig = pointcloud_1.copy()
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_1(pointcloud_1)
        point_t2 = trans_2(pointcloud_2)

        # pointcloud = (pointcloud_orig, point_t1, point_t2)
        pointcloud = (point_t1, point_t2)
        return pointcloud, render_img # render_img_list

    def __len__(self):
        return len(self.data)


def shapenet_map_key_to_class_name(key):
    class_dict = {'04379243': 'table', 
                  '03593526': 'jar', 
                  '04225987': 'skateboard', 
                  '02958343': 'car', 
                  '02876657': 'bottle', 
                  '04460130': 'tower', 
                  '03001627': 'chair', 
                  '02871439': 'bookshelf', 
                  '02942699': 'camera', 
                  '02691156': 'airplane', 
                  '03642806': 'laptop', 
                  '02801938': 'basket', 
                  '04256520': 'sofa', 
                  '03624134': 'knife', 
                  '02946921': 'can', 
                  '04090263': 'rifle', 
                  '04468005': 'train', 
                  '03938244': 'pillow', 
                  '03636649': 'lamp', 
                  '02747177': 'trash bin', 
                  '03710193': 'mailbox', 
                  '04530566': 'watercraft', 
                  '03790512': 'motorbike', 
                  '03207941': 'dishwasher', 
                  '02828884': 'bench', 
                  '03948459': 'pistol', 
                  '04099429': 'rocket', 
                  '03691459': 'loudspeaker', 
                  '03337140': 'file cabinet', 
                  '02773838': 'bag', 
                  '02933112': 'cabinet', 
                  '02818832': 'bed', 
                  '02843684': 'birdhouse', 
                  '03211117': 'display', 
                  '03928116': 'piano', 
                  '03261776': 'earphone', 
                  '04401088': 'telephone', 
                  '04330267': 'stove', 
                  '03759954': 'microphone', 
                  '02924116': 'bus', 
                  '03797390': 'mug', 
                  '04074963': 'remote', 
                  '02808440': 'bathtub', 
                  '02880940': 'bowl', 
                  '03085013': 'keyboard', 
                  '03467517': 'guitar', 
                  '04554684': 'washer', 
                  '02834778': 'bicycle', 
                  '03325088': 'faucet', 
                  '04004475': 'printer', 
                  '02954340': 'cap'}
    
    return class_dict.get(key, None)

class ShapeNetRender_for_CrossPoint(Dataset):
    def __init__(self, img_transform = None, n_imgs = 1, exclusion_list=[]):
        self.data = load_shapenet_data(exclusion_list)
        print("Crosspoint - self.data length:",len(self.data))
        self.transform = img_transform
        self.n_imgs = n_imgs
    
    def __getitem__(self, item):
        pcd_path = self.data[item]
        tok = self.data[item].split(os.sep)
        label = shapenet_map_key_to_class_name(tok[len(tok)-2])
        imgs_rendered = get_render_imgs(pcd_path)
        if len(imgs_rendered) < 2:
          print(f'pcd_path: {pcd_path}')
        render_img_path = random.choice(imgs_rendered)
        render_img_path_token_list = render_img_path.split(os.sep)
        img_name = render_img_path_token_list[len(render_img_path_token_list)-1]

        if self.n_imgs > 1:
            render_img2_path = random.choice(imgs_rendered)
            render_img2_path_token_list = render_img2_path.split(os.sep)
            img2_name = render_img2_path_token_list[len(render_img2_path_token_list)-1]

            while img2_name == img_name: #ensure that the two images are different
                render_img2_path = random.choice(imgs_rendered)
                render_img2_path_token_list = render_img2_path.split(os.sep)
                img2_name = render_img2_path_token_list[len(render_img2_path_token_list)-1]

        render_img = Image.open(render_img_path).convert('RGB')
        render_img = self.transform(render_img)  #.permute(1, 2, 0)

        if self.n_imgs > 1:
            render_img_2 = Image.open(render_img2_path).convert('RGB')
            render_img_2 = self.transform(render_img_2)  #.permute(1, 2, 0)

        pointcloud_1 = load_ply(self.data[item])
        pointcloud_2 = load_ply(self.data[item])
        point_t1 = trans_1(pointcloud_1)
        point_t2 = trans_2(pointcloud_2)

        pointcloud = (point_t1, point_t2)
        if self.n_imgs > 1:
            return pointcloud, render_img, render_img_2, label
        else:
            return pointcloud, render_img, label

    def __len__(self):
        return len(self.data)


class ModelNet40SVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_modelnet_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
    
class ScanObjectNNSVM(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_ScanObjectNN(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

# ------------------- MODELNET MBT -----------------
def read_off(filename):
    num_select = 1024
    # num_select = 2048
    f = open(filename)
    f.readline()  # ignore the 'OFF' at the first line
    f.readline()  # ignore the second line
    All_points = []
    selected_points = []
    while True:
        new_line = f.readline()
        x = new_line.split(' ')
        if x[0] != '3':
            A = np.array(x[0:3], dtype='float32')
            All_points.append(A)
        else:
            break
    # if the numbers of points are less than 2000, extent the point set
    if len(All_points) < (num_select + 3):
        return None
    # take and shuffle points
    index = np.random.choice(len(All_points), num_select, replace=False)
    for i in range(len(index)):
        selected_points.append(All_points[index[i]])

    return np.array(selected_points)  # return N*3 array


def load_modelnet_data_mbt(modelnet_exclusion_list):
  #Return point cloud tuple, image(s) and label
  BASE_DIR = './drive/MyDrive/VP_CrossPoint_2023/data/modelnet40_MBT'
  all_classes = os.listdir(BASE_DIR)
  set_differences = []
  all_data = []
  all_imgs_path = []
  all_label = []
  for elem in all_classes:
      if elem not in modelnet_exclusion_list:
          set_differences.append(elem)
          
  for obj in set_differences:
    DATA_DIR = os.path.join(BASE_DIR, obj)
    print(DATA_DIR)
    print("Start "+obj)
    for dir in tqdm(os.listdir(DATA_DIR)):
        if os.path.isdir(os.path.join(DATA_DIR, dir)):
            if dir != "Store":
                subdir = os.path.join(DATA_DIR, dir)
                for file in os.listdir(subdir):
                    if file.endswith(".off"):
                        file_path = os.path.join(subdir, file)
                        current_off = read_off(file_path)
                        if current_off is not None:
                          all_data.append(current_off)
                          all_label.append(obj)
                          all_imgs_path.append(subdir)
    print("End "+obj)    
                    
                        
  return all_data,all_imgs_path,all_label


def get_modelnet_render_imgs_mbt(pcd_path):
    path_lst = pcd_path.split(os.sep)
    
    DIR = '/'.join(path_lst)
    img_path_list = glob.glob(os.path.join(DIR, '*.png'))
    
    return img_path_list

class ModelNet40SVM_MBT(Dataset):
    def __init__(self, img_transform=None, modelnet_exclusion_list=[], n_imgs=1):
        self.data, self.subdir, self.labels = load_modelnet_data_mbt(modelnet_exclusion_list)
        self.transform = img_transform
        self.n_imgs = n_imgs
        
    def __getitem__(self, item):
        imgs_rendered = get_modelnet_render_imgs_mbt(self.subdir[item])

        render_img_path = random.choice(imgs_rendered)
        render_img_path_token_list = render_img_path.split(os.sep)
        img_name = render_img_path_token_list[len(render_img_path_token_list)-1]
        if self.n_imgs > 1:
            render_img2_path = random.choice(imgs_rendered)
            render_img2_path_token_list = render_img2_path.split(os.sep)
            img2_name = render_img2_path_token_list[len(render_img2_path_token_list)-1]

            while img2_name == img_name: #ensure that the two images are different
                render_img2_path = random.choice(imgs_rendered)
                render_img2_path_token_list = render_img2_path.split(os.sep)
                img2_name = render_img2_path_token_list[len(render_img2_path_token_list)-1]
        
        pointcloud = trans_val(self.data[item])
        label = self.labels[item]
        render_img = Image.open(render_img_path)
        
        if self.transform is not None:
            render_img = self.transform(render_img)  #.permute(1, 2, 0)

        if self.n_imgs > 1:
            render_img_2 = Image.open(render_img2_path)
            if self.transform is not None:
                render_img_2 = self.transform(render_img_2)  #.permute(1, 2, 0)

        if self.n_imgs > 1:
            return pointcloud, render_img, render_img_2, label
        else:
            return pointcloud, render_img, label

    def __len__(self):
        return len(self.data)