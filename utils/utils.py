import torch
import torch.nn as nn

import time
import datetime
class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, preds, labels, masks):
        seq_length = preds.size(1)
        batch_size = preds.size(0)
        total_loss = torch.tensor(0.0)
        labels = labels[:, :seq_length]
        masks = masks[:, :seq_length]
        for t in range(seq_length):
            current_loss = self.xe_loss(preds[:, t], labels[:, t])
            current_loss *= masks[:, t]
            total_loss += torch.sum(current_loss)
        return total_loss / batch_size

def set_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group['lr'] = learning_rate

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.clamp_(-grad_clip, grad_clip)

def get_duration(start):
    now = time.time()
    seconds = now - start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    pattern = "%02d hour(s): %02d minutes: %02d seconds" %\
              (h, m, s)
    if d > 0:
        pttern = ("%d days: " + pattern)%(d)
    return pattern

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[str(ix.item())]
            else:
                break
        out.append(txt)
    return out
if __name__ == "__main__":
    start = time.time()
    time.sleep(10)
    print("cost", get_duration(start))