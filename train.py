import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

import time
import os
import pickle
import tensorboardX

import opts
from models.encoder import Encoder
from models.decoder import Decoder
import utils.utils as utils
from utils.data_loader import get_loader
import eval_utils

def train(opt):
    loader = get_loader(opt, 'train')
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    summry_writer = tensorboardX.SummaryWriter()

    infos = {}
    histories = {}
    if opt.start_from is not None:
        infos_path = os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl')
        histories_path = os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')
        # open infos and check if models are compatible
        with open(infos_path, 'rb') as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same = ['hidden_size']
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme],\
                "Command line argument and saved model disagree on %s"%(checkme)
        if os.path.isfile(histories_path):
            with open(histories_path, 'rb') as f:
                histories = pickle.load(f)

    iteration = infos.get('iter', 0)
    current_epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})

    if opt.load_best_score == 1:
        best_val_score = infos.get("best_val_score", None)

    encoder = Encoder()
    decoder = Decoder(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion =  utils.LanguageModelCriterion().to(device)
    optimizer = optim.Adam(decoder.parameters(), lr=opt.learning_rate,
                           weight_decay=opt.weight_decay)


    if vars(opt).get('start_from', None) is not None:
        optimizer_path = os.path.join(opt.start_from, 'optimizer.pth')
        optimizer.load_state_dict(torch.load(optimizer_path))

    total_step = len(loader)
    start = time.time()
    for epoch in range(current_epoch, opt.max_epochs):
        if epoch > opt.learning_rate_decay_start and \
            opt.learning_rate_decay_start >= 0:
            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
            deccay_factor = opt.learning_rate_decay_rate ** frac
            opt.current_lr = opt.learning_rate * deccay_factor
            utils.set_lr(optimizer, opt.current_lr)
            print("learing rate change form {} to {}".format(opt.learning_rate,
                                                             opt.current_lr))
        else:
            opt.current_lr = opt.learning_rate
        for i, data in enumerate(loader, iteration):
            if i > total_step - 1:
                iteration = 0
                break
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

            with torch.no_grad():
                features = encoder(imgs)
            preds = decoder(features, labels)

            loss = criterion(preds, labels[:, 1:], masks[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss = loss.item()

            print("iter: {}/{} (epoch {}), train loss = {:.3f}, time/batch = {}"\
                 .format(i, total_step, epoch, train_loss, utils.get_duration(start)))

            log_iter = i + epoch*total_step
            # write training loss summary
            if (i % opt.losses_log_every) == 0:
                summry_writer.add_scalar('train_loss', train_loss, log_iter)
                summry_writer.add_scalar('learning_rate', opt.current_lr, log_iter)

            # make evaluation on validation set, and save model
            if (i % opt.save_checkpoint_every == 0):
                #eval model
                eval_kwargs = {'split': 'val',
                               'dataset': opt.input_json}
                eval_kwargs.update(vars(opt))
                val_loss,\
                predictions,\
                lang_stats = eval_utils.eval_split(encoder, decoder, criterion,
                                                   opt, eval_kwargs)
                summry_writer.add_scalar('valaidation loss', val_loss, log_iter)
                if lang_stats is not None:
                    for metric, score in lang_stats.items():
                        summry_writer.add_scalar(metric, score, log_iter)
                val_result_history[i] = {"loss": val_loss,
                                         "lang_stats": lang_stats,
                                         "predictions": predictions}

                if opt.language_eval == 1:
                    current_score = lang_stats['CIDEr']
                else:
                    current_score = -val_loss

                best_flag = False
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                if not os.path.exists(opt.checkpoint_path):
                    os.makedirs(opt.checkpoint_path)
                checkpoint_ptah = os.path.join(opt.checkpoint_path, 'model.pth')
                torch.save(decoder.state_dict(), checkpoint_ptah)
                print("model saved to {}".format(checkpoint_ptah))

                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                # Dump miscalleous informations
                infos['iter'] = i+1
                infos['epoch'] = epoch
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.ix_to_word

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                infos_path = os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'.pkl')
                histories_path = os.path.join(opt.checkpoint_path, 'histories_'+opt.id+'.pkl')
                with open(infos_path, 'wb') as f:
                    pickle.dump(infos, f)
                print("infos saved into {}".format(infos_path))

                with open(histories_path, 'wb') as f:
                    pickle.dump(histories, f)
                print('histories saved into {}'.format(histories_path))
                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(decoder.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        pickle.dump(infos, f)

    summry_writer.close()
if __name__ == "__main__":
    opt = opts.parse_opts()
    train(opt)
