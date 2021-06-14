import os
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np
import pickle
from .make_border_map import MakeBorderMap
from .make_seg_detector_data import MakeSegDetectorData
from utils import resize_with_coordinates, box2seg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.FloatTensor)

def get_dataloader(conf):
    return DataLoader(DetectionData(conf),batch_size = conf['batch_size'],shuffle=True,collate_fn = collator)

class DetectionData(Dataset):
    def __init__(self,conf):
        super(DetectionData,self).__init__()
        if not os.path.exists(conf['data_path']):
            raise
        self.data_path = conf['data_path']
        self.data_list = list()
        
        self.box_folder = 'orig_boxes'
        self.img_folder = 'orig_texts'
        
        self.height = 1024
        self.width = 768

        for filename in os.listdir(os.path.join(self.data_path,self.box_folder)):
            file = filename.split('.')[0]
            self.data_list.append(file)
        
        self.border_map = MakeBorderMap()
        self.seg_map = MakeSegDetectorData()
        
    def __getitem__(self,idx):
        image = Image.open(os.path.join(self.data_path,\
                                        self.img_folder,\
                                        self.data_list[idx]+'.png'))
        img = np.asarray(image)/255
        
        with open(os.path.join(self.data_path,\
                               self.box_folder,\
                               self.data_list[idx]+'.pkl'),'rb') as f:
            boxes = pickle.load(f)
        
        for i,word in enumerate(boxes):
            for j,point in enumerate(boxes[i]['box']):
                temp = point[0]
                boxes[i]['box'][j][0] = point[1]
                boxes[i]['box'][j][1] = temp
            exchange = boxes[i]['box'][2]
            boxes[i]['box'][2] = boxes[i]['box'][3]
            boxes[i]['box'][3] = exchange
            
        gt = np.concatenate([i['box'][np.newaxis,:,:]for i in boxes])
        re_img, re_bbox = resize_with_coordinates(img,self.width,self.height,gt)
        
        ignore_tags = np.ones(re_bbox.shape[0], dtype=np.uint8)
        re_y, re_seg, re_mask = box2seg(re_img,re_bbox)
        
        data = dict(image=re_img,polygons=re_bbox,ignore_tags=ignore_tags)
        data = self.border_map(data)
        data = self.seg_map(data)
        
        return data
   
    def __len__(self):
        return len(self.data_list)
    
def collator(batch):
    new_batch = dict(image=np.zeros(batch[0]['image'].shape)[np.newaxis,:,:,:],\
                     polygons=[],ignore_tags=[],\
                     thresh_map=np.zeros(batch[0]['thresh_map'].shape)[np.newaxis,:,:],\
                     thresh_mask=np.zeros(batch[0]['thresh_mask'].shape)[np.newaxis,:,:],\
                     gt=np.zeros(batch[0]['gt'].shape)[np.newaxis,:,:,:],\
                     mask=np.zeros(batch[0]['mask'].shape)[np.newaxis,:,:])
    
    for i,row in enumerate(batch):
        new_batch['image'][i] = row['image']
        new_batch['polygons'].append(row['polygons'])   
        new_batch['ignore_tags'].append(row['ignore_tags'])   
        new_batch['thresh_map'][i] = row['thresh_map']        
        new_batch['thresh_mask'][i] = row['thresh_mask']        
        new_batch['gt'][i] = row['gt']        
        new_batch['mask'][i] = row['mask']                
        
    new_batch['image'] = torch.from_numpy(new_batch['image']).to(device)
    new_batch['thresh_map'] = torch.from_numpy(new_batch['thresh_map']).to(device)    
    new_batch['thresh_mask'] = torch.from_numpy(new_batch['thresh_mask']).to(device)    
    new_batch['gt'] = torch.from_numpy(new_batch['gt']).to(device)    
    new_batch['mask'] = torch.from_numpy(new_batch['mask']).to(device)
    
    return new_batch