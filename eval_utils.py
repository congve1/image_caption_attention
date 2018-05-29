import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
import random
import string
import time
import os
import sys
import utils.utils as utils
from utils.data_loader import get_loader

def eval_split(encoder, decoder, crit, opt, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images',
                                 eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(opt, split)
    decoder.eval()
    with torch.no_grad():
        loss = 0
        loss_sum = 0
        loss_evals = 1e-8
        predictions = []
        total_step = len(loader)
        start = time.time()
        for i, data in enumerate(loader, 0):
            transform = transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))
            imgs = []
            for k in range(data['imgs'].shape[0]):
                img = torch.tensor(data['imgs'][k], dtype=torch.float)
                img = transform(img)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0).to(device)
            labels = torch.tensor(data['labels'].astype(np.int32),
                                  dtype=torch.long).to(device)
            masks = torch.tensor(data['masks'], dtype=torch.float).to(device)

            features = encoder(imgs)
            seqs = decoder(features, labels)
            loss = crit(seqs, labels[:, 1:], masks[:, 1:])
            loss_sum += loss
            loss_evals += 1
            seq, _ = decoder.sample(features)

            sents = utils.decode_sequence(loader.ix_to_word,
                                          seq[torch.arange(loader.batch_size,dtype=torch.long)*
                                              loader.seq_per_img])

            print("batch [{} / {}] cost: {}".format(i, total_step, utils.get_duration(start)))
            for k, sent in enumerate(sents):
                entry = {"image_id":data['infos'][k]['id'],
                         "caption": sent}
                predictions.append(entry)

                if verbose:
                    print("image: %s: %s"%(entry['image_id'], entry['caption']))

            if num_images >= 0 and (i+1)*loader.batch_size >= num_images:
                break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)
    decoder.train()
    return loss_sum/loss_evals, predictions, lang_stats


def language_eval(dataset, predictions, model_id, split):
    import sys
    if 'coco' in dataset:
        sys.path.append("coco-caption")
        annFile = 'coco-caption/annotations/captions_val2014.json'

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    json.encoder.FLOAT_REPR = lambda o: format(0, '.3f')

    if not os.path.exists('eval_results'):
        os.makedirs('eval_results')

    cache_path = os.path.join('eval_results', model_id+'_'+split+'.json')
    coco = COCO(annFile)
    valids = coco.getImgIds()
    preds_filt = [p for p in predictions if p['image_id'] in valids]
    print('using %d/%d predictions'%(len(preds_filt), len(predictions)))
    json.dump(preds_filt, open(cache_path, 'w'))

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    #create output dict
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    imgToEval = cocoEval.imgToEval

    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall':out, 'imgToEval': imgToEval}, outfile)

    return out
