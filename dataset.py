import numpy as np
import os 
import sys 
import glob 
import medpy.io as io
import pickle as plk
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool

class DataSet:
    image_height = 512
    image_width = 512
    num_classes = 2


    def __init__(self,image_dir,bbox_path,label_path=None):
        '''
        训练集、验证集、测试集应放置在不同的文件夹下，并且各自使用独立的DataSet实例
        Args:
            image_dir: 
            bbox_path:
            label_path: Set "None" for test dataset
        '''
        self.image_paths = glob.glob(image_dir+'/*.nii')
        self.image_num = len(self.image_paths)
        assert self.image_num != 0, '''There're no images in "image_dir" !'''
        with open(bbox_path,'rb') as f:
            self.bbox = plk.load(f)
        if label_path != None:
            self.label = pd.read_csv(label_path)
            self.label = self.label.set_index('id')


    def patch_based_random_batch(self,image_dtype,label_dtype,batch_size,width=224,height=224,depth=24):
        '''
        基于patch的随机批次，直接使用CT图像的标签作为patch的标签，用于基于patch的网络训练
        使用多线程以加快数据加载速度
        Args:
            image_dtype: images' data type you need
            label dtype: labels' data type you need
            batch_size: how many patches in a batch
            width: image width you need
            height: image height you need
            depth: image depth you need
        Returns:
            X: 5d numpy array, shape of [batch_size,height,width,depth,channel]("channel" = 1),
             dtype of "image_dtype"
            Y: 1d numpy array, shape of [batch_size], dtype of "label_dtype"
        '''
        #---------Get random image index-----------
        np.random.shuffle(self.image_paths)
        random_index = np.random.randint(low=0,high=self.image_num-1,size=batch_size)

        #--------Base function for getting one patch from a CT Volume in bounding box--------
        def base_func(param_list):
            image_path = param_list[0]
            image_name = image_path.split('/')[-1].split('.')[0]
            image,head = io.load(image_path)
            #----------Bounding box----------------------------
            bbox_coordinate_min = []
            bbox_coordinate_max = []
            for i in range(3):
                bbox_coordinate_min.append(self.bbox[image_name][2*i])
                bbox_coordinate_max.append(self.bbox[image_name][2*i+1])
            #----------Extrace one patch from bbox------------------------------------------------------------
            # patch_size is [height,width,depth], a patch's coordinate_min will be randomly selected
            # from the space of [bbox_coordinate_min, bbox_coordinate_max - patch_size] to ensure 
            # that the patch extracted is totally in the bbox
            patch_coordinate_min = []
            patch_coordinate_max = []
            patch_size = [height,width,depth]
            for i in range(3):
                patch_coordinate_min.append(np.random.randint(bbox_coordinate_min[i], bbox_coordinate_max[i]-patch_size[i]))
                patch_coordinate_max.append(patch_coordinate_min[i] + patch_size[i])
            patch = image[patch_coordinate_min[0]:patch_coordinate_max[0],\
                          patch_coordinate_min[1]:patch_coordinate_max[1],\
                          patch_coordinate_min[2]:patch_coordinate_max[2] ]
            #------------------Get label-----------------
            label = int(self.label.loc[image_name])

            return patch,label

        patch,label = base_func([self.image_paths[0]])

        #-------Load Images with multiple threads-------------------
        thread_num = int(np.ceil(batch_size / 4))
        param_list = []
        for i in random_index:
            param_list.append([self.image_paths[i]])
        pool = ThreadPool(processes=thread_num)
        result_list = pool.map(func=base_func,iterable=param_list)
        pool.close()
        pool.join()

        #--------Get data and return----------------------------------------
        X = np.zeros(shape=[batch_size,height,width,depth],dtype=image_dtype)
        Y = np.zeros(shape=[batch_size],dtype=label_dtype)
        for i in range(batch_size):
            X[i,:,:,:] = result_list[i][0]
            Y[i] = result_list[i][1]

        X = X[:,:,:,:,np.newaxis]
        return X,Y
        

    def volume_base_random_batch(self):
        '''
        用于基于volume的分类
        '''
        pass

    def sliding_window_based_batch(self):
        '''
        用于最终提交结果的测试集的采样和交叉验证集的采样
        '''
        pass


if __name__ == "__main__":
    dataset = DataSet('./TrainData','./BBOX.pkl','./train_label.csv')
    X,Y = dataset.patch_based_random_batch('float32','int16',12)
    print(X.shape)
    print(Y.shape)
    

        