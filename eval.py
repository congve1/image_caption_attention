import torch
import torch.nn as nn

import argparse
import json
import pickle

import models
import eval_utils
import utils.utils as utils
from utils.data_loader import get_loader
import opts

def main(opt):
    with open(opt.infos_path, 'rb') as f:
        infos = pickle.load(f)

    #override and collect parameters
    if len(opt.input_h5) == 0:
        opt.input_h5 = infos['opt'].input_h5

    if len(opt.input_json) == 0:
        opt.input_json = infos['opt'].input_json

    if opt.batch_size == 0:
        opt.batch_size = infos['opt'].batch_size

    if len(opt.id) == 0:
        opt.id = infos['opt'].id
    ignore = ['id', 'batch_size', 'beam_size', 'strat_from', 'language_eval']

    for key, value in vars(infos['opt']).items():
        if key not in ignore:
            if key in vars(opt):
                assert vars(opt)[key] == vars(infos['opt'])[key],\
                key+" option not consistent"
            else:
                vars(opt).update({key: value})
    vocab = infos['vocab']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = models.encoder.Encoder().to(device)
    decoder = models.decoder.Decoder(opt).to(device)
    decoder.load_state_dict(torch.load(opt.model, map_location=str(device)))
    encoder.eval()
    decoder.eval()
    criterion = utils.LanguageModelCriterion().to(device)
    if len(opt.image_folder) == 0:
        loader = get_loader(opt, 'test')
        loader.ix_to_word = vocab
        loss, split_predictions, lang_stats = \
        eval_utils.eval_split(encoder, decoder, criterion, opt, vars(opt))
        print('loss: ', loss)
        print(lang_stats)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Input paths
    parser.add_argument('--model', type=str, default="")
    parser.add_argument('--infos_path', type=str, default="")

    #basic option
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=-1)
    parser.add_argument('--language_eval', type=int, default=0)

    parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
    parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
    parser.add_argument('--input_json', type=str, default='',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
    parser.add_argument('--input_h5', type=str, default='')
    parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
    # misc
    parser.add_argument('--id', type=str, default='',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

    opt = parser.parse_args()
    main(opt)