import json
import os
import random

import torch
import torchvision.transforms as transform
import torch.utils.data as data
import h5py
import numpy as np

class COCODataset(data.Dataset):
    def __init__(self, opt, split):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img
        self.transform = transform
        self.batch_size = opt.batch_size
        #load json file which contain additional information about dataset
        print("Dataset Loading json file", opt.input_json)
        self.info = json.load(open(opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print("Vocabulary size is: ", self.vocab_size)
        #load h5py file
        self.h5_file = h5py.File(opt.input_h5, 'r')
        seq_size = self.h5_file['labels'].shape
        self.seq_length = seq_size[1]
        print("max sequence length is", self.seq_length)
         # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_file['label_start_ix'][:]
        self.label_end_ix = self.h5_file['label_end_ix'][:]
        self.num_images = self.label_start_ix.shape[0]
        self.split = split
        self.split_ix = {'train': [], 'val': [], 'test':[]}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: #restval
                self.split_ix['train'].append(ix)
        print("assigned %d images to split train"%(len(self.split_ix['train'])))
        print("assigned %d images to split val"%(len(self.split_ix['val'])))
        print("assigned %d images to split test"%(len(self.split_ix['test'])))

    def __getitem__(self, index):
        masks = np.zeros([self.seq_per_img, self.seq_length+2], dtype="float32")
        labels = np.zeros([self.seq_per_img, self.seq_length+2], dtype="uint32")
        #print(index)
        ix = self.split_ix[self.split][index]
        #print(ix)
        img = self.info['images'][ix]
        #print(img)
        l_start_ix = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        l_end_ix = self.label_end_ix[ix] - 1
        ncap = l_end_ix - l_start_ix + 1
        if ncap < self.seq_per_img:
            # need to subsample
            seq = np.zeros([self.seq_per_img, self.seq_length], dtype="uint32")
            seq_lengths = np.zeros([self.seq_per_img], dtype='uint32')
            for q in range(self.seq_per_img):
                ixl = random.randint(l_start_ix, l_end_ix)
                seq[q, :] = self.h5_file['labels'][ixl, :self.seq_length]


        else:
            ixl = random.randint(l_start_ix, l_end_ix - self.seq_per_img + 1)
            seq = self.h5_file['labels'][ixl:ixl+self.seq_per_img,
                                        :self.seq_length]
        labels[:, 1:self.seq_length+1] = seq
        nonzeros = np.array([(s != 0).sum() + 2 for s in seq])
        for i, row in enumerate(masks):
            row[:nonzeros[i]] = 1.
        tmp_img = self.h5_file['images'][ix]
        out_size = (self.opt.crop_size, self.opt.crop_size)
        tmp_img = self._random_crop(tmp_img, out_size)
        #print(ix)
        imgs = np.stack([tmp_img for _ in range(self.seq_per_img)])
        imgs = (imgs.astype(np.float)) / 255. #normalize img to 0-1.
        info_dict = {}
        info_dict['id'] = img['id']
        #print(img['id'])
        info_dict['file_path'] = img['file_path']
        data = {}
        data['imgs'] = imgs
        data['labels'] = labels
        data['masks'] = masks
        data['info_dict'] = info_dict
        return data


    def __len__(self):
        return len(self.split_ix[self.split])

    def _random_crop(self, img, output_size):
        w, h = img.shape[2], img.shape[1]
        th, tw = output_size
        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)
        t_img = img[:, i:i+th, j:j+tw]
        return t_img



def collate_fn(datas):
    final_data_batch = {}
    final_data_batch['imgs'] = np.concatenate([data['imgs'] for data in datas], axis=0)
    final_data_batch['labels'] = np.concatenate([data['labels'] for data in datas], axis=0)
    final_data_batch['masks'] = np.concatenate([data['masks'] for data in datas], axis=0)
    final_data_batch['infos'] = [data['info_dict'] for data in datas]
    return final_data_batch

def get_dataset(opt, split):
    return COCODataset(opt, split)

def get_loader(opt, split):
    dataset = get_dataset(opt, split)
    loader = data.DataLoader(dataset, opt.batch_size, True, collate_fn=collate_fn,
                             drop_last=True)
    loader.vocab_size = dataset.vocab_size
    loader.seq_length = dataset.seq_length
    loader.ix_to_word = dataset.ix_to_word
    loader.seq_per_img = dataset.seq_per_img
    return loader