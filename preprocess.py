import numpy as np
import os 
import sys
import glob
import medpy.io as io
import cv2
import time
import pickle as pkl


def generate_mask(origin_image_paths,save_dir):
    '''
    从原图生成mask，mask中背景为0，人体为1
    Args:
        origin_image_paths:
        save_dir:
    Returns:
        None
    '''
    assert os.path.exists(save_dir),"Save dir doesn't exist"
    image_paths = origin_image_paths

    for image_path in image_paths:
        image,head = io.load(image_path)
        image[image<0] = 0
        image[image>200] = 200
        image = image.astype('uint8')
        kernel = np.ones((10,10),np.uint8)
        mask = np.zeros(image.shape,dtype=np.uint8)

        temp = np.zeros((image.shape[0],image.shape[1]),np.uint8)
        for i in range(image.shape[2]):
            temp = image[:,:,i]
            #开运算
            temp = cv2.morphologyEx(temp,cv2.MORPH_OPEN,kernel)
            #二值化
            temp[temp>0] = 1
            #膨胀操作
            temp = cv2.dilate(temp,kernel,iterations=1)
            mask[:,:,i] = temp

        save_path = save_dir + '/' + image_path.split('/')[-1] + '.nii'
        io.save(mask,save_path,head)

def clip_and_multiply_mask(origin_image_paths,mask_dir,save_dir):
    '''
    原图裁剪到[-200,250]范围内（有意义的像素点大多分布在这个范围内），然后乘以mask以移除非ROI区域
    Args:
        origin_image_paths:
        mask_dir:
        save_dir:
    Returns:
        None
    '''
    assert os.path.exists(mask_dir),"Mask dir doesn't exist"
    assert os.path.exists(save_dir),"Save dir doesn't exist"
    image_paths = origin_image_paths
    mask_paths = glob.glob(mask_dir+'/*')
    assert len(image_paths) == len(mask_paths),"Number of images and masks doesn't match, delete masks and generate again!"

    image_paths.sort()
    mask_paths.sort()
    for image_path,mask_path in zip(image_paths,mask_paths):
        assert image_path.split('/')[-1] == mask_path.split('/')[-1].split('.')[0],\
        "Image and mask doesn't match"
        image,head = io.load(image_path)
        mask,_ = io.load(mask_path)
        image = np.clip(image,0,200)
        image = np.multiply(image,mask)
        save_path = save_dir + '/' + mask_path.split('/')[-1]
        io.save(image,save_path,head)

def find_bbox(mask_dir,bbox_save_path):
    '''
    有效体积占整张CT图像50%左右，把计算放在有效体积上可以提高计算效率，也有利于减少
    假阳性（标签为肿瘤的CT图像的有效体积内，选取的局部体积包含肿瘤细胞的概率更大）
    本函数分别在x,y,z轴上，采用阈值的方法寻找有效体积的坐标
    bbox保存格式： python dict, key为CT图像编号，value为python list [x_min,x_max,y_min,y_max,z_min,z_max]
    Args:
        mask_dir:
        bbox_save_path:
    Returns:
        None
    '''
    assert os.path.exists(mask_dir),"Mask dir doesn't exist!"
    mask_paths = glob.glob(mask_dir+'/*')

    roi_ranges = {}
    for mask_path in mask_paths:
        mask_name = mask_path.split('/')[-1].split('.')[0]
        mask,head = io.load(mask_path)
        volume = mask.shape[0] * mask.shape[1] * mask.shape[2]
        total_voxel = np.count_nonzero(mask)
        pencent_avg = total_voxel / volume
        roi_range = []

        #------Top view, calculate x-axis ROI range--------
        top_area = mask.shape[1] * mask.shape[2]
        mark1 = mark2 = 0
        for i in range(mask.shape[0]):
            pixel = np.count_nonzero(mask[i,:,:])
            pencent = pixel / top_area
            if pencent < pencent_avg*0.4:
                mark2 = mark1
                mark1 = i
                if mark1 > (mark2 + 100):
                    roi_range.append(mark2)
                    roi_range.append(mark1)
                    break
        
        #------Side view, calculate y-axis ROI range-------
        side_area = mask.shape[0] * mask.shape[2]
        mark1 = mark2 = 0
        for i in range(mask.shape[1]):
            pixel = np.count_nonzero(mask[:,i,:])
            pencent = pixel / side_area
            if pencent < pencent_avg*0.4:
                mark2 = mark1
                mark1 = i
                if mark1 > (mark2 + 100):
                    roi_range.append(mark2)
                    roi_range.append(mark1)
                    break

        #-----Front view, calculate z-axis ROI range-------
        front_area = mask.shape[0] * mask.shape[1]
        mark1 = mark2 = 0
        for i in range(mask.shape[2]):
            pixel = np.count_nonzero(mask[:,:,i])
            pencent = pixel / front_area
            if pencent < pencent_avg*0.9:
                mark2 = mark1
                mark1 = i
                if mark1 > (mark2 + 10):
                    roi_range.append(mark2)
                    roi_range.append(mark1)
                    break

        roi_ranges[mask_name] = roi_range

    with open(bbox_save_path,'wb') as f:
        pkl.dump(roi_ranges,f)

         

if __name__ == "__main__":
    
    # origin_dirs = ['dir1','dir2','dir3']
    origin_dirs = ['./TrainOrigin']


    origin_image_paths = []
    for origin_dir in origin_dirs:
        origin_image_paths += glob.glob(origin_dir+'/*')
    # 生成Mask
    generate_mask(origin_image_paths,'./Mask')
    # 处理原图
    clip_and_multiply_mask(origin_image_paths,'./Mask','./TrainData')
    # 寻找Bounding box
    find_bbox('./Mask','BBOX.pkl')
    print('Finished.')

    
