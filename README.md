# image_caption_attention
## Introduction
- This code is a PyTorch implementation of [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044).
- Use Resnet.
- Use [Karpathy's train-val-test split](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
## References

Author's original theano implementation: <https://github.com/kelvinxu/arctic-captions>

Yunjey's tensorflow implementation: <https://github.com/yunjey/show-attend-and-tell>

neuraltalk2:<https://github.com/karpathy/neuraltalk2>

ruotianluo's image captioning code: <https://github.com/ruotianluo/ImageCaptioning.pytorch>
## Getting started

### Dependencies

To use this code, you need to install:
- Python3.6
- PyTorch 0.4 along with torchvision
- matplotlib
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
- numpy
- pycocotools
- h5py

You can use pip to install pycocotools directly or compile from source code.

### Prepare Dataset
First, download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images.
Then, download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and extract `dataset_coco.json` from zip file into `data/`.
Then, invode `scripts/prepo.py` script, which will create a dataset(an hdf5 file and a json file).
```bash
$ python scripts/prepo.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk.h5 --word_count_threshold 5 --images_root data
```
Warning: the prepro script will fail with the default MSCOCO data because one of their images is corrupted. See this [issue](https://github.com/karpathy/neuraltalk2/issues/4) for the fix, it involves manually replacing one image in the dataset.

### Start training
```bash
$ python train.py --id st --input_json data/cocotalk.json --input_h5 data/cocotalk.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --checkpoint_path log_st --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 25
```
