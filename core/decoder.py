import torch
import torch.nn as nn
import torch.nn.functional as F
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.att_feat_size = opt.att_feat_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size
        self.vocab_size = opt.vocab_size
        self.seq_length = opt.seq_length
        self.drop_prob_lm = opt.drop_prob_lm

        self.init_c = nn.Linear(self.att_feat_size, self.hidden_size)
        self.init_h = nn.Linear(self.att_feat_size, self.hidden_size)

        self.bn = nn.BatchNorm1d(self.att_feat_size)

        self.embedding =  nn.Embedding(self.vocab_size+1, self.embedding_size)
        self.core = AttCore(opt)
        self.logit = nn.Linear(self.embedding_size, self.vocab_size+1)

    def forward(self, features, seq):
        batch_size = features.size(0)
        outputs = []

        #change features shape from BxDxL to BxLxD
        features = self.bn(features).permute(0, 2, 1)

        h, c = self.init_state(features)

        for t in range(seq.size(1)-1):
            # break if all the sequences end
            if t >= 1 and seq[:, t].sum() == 0:
                break
            it = seq[:, t].clone()
            xt = self.embedding(it)
            output, (h, c) = self.core(xt, features, (h, c))
            logit = self.logit(output)
            outputs.append(logit)
        return torch.stack(outputs, dim=1)

    def sample(self, features):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features = self.bn(features).permute(0, 2, 1)
        beam_size = self.opt.beam_size
        if beam_size > 1:
            return self.beam_sample(features)
        batch_size = features.size(0)
        h, c = self.init_state(features)
        seq = []
        seq_log_probs = []
        for t in range(self.seq_length+1):
            if t == 0: # <bos>
                it = torch.zeros(batch_size, dtype=torch.long)
                it = it.to(device)
            else:
                sample_log_probs, it = torch.max(logprobs, 1)
                it = it.view(-1).long()
            xt = self.embedding(it)
            if t >= 1:
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it)
                seq_log_probs.append(sample_log_probs.view(-1))
            output, (h, c) = self.core(xt, features, (h, c))
            logprobs = F.log_softmax(self.logit(output), dim=1)
        return torch.stack(seq, 1), torch.stack(seq_log_probs, 1)


    def beam_sample(self, features):
        batch_size = features.size(0)
        beam_size = self.opt.beam_size
        seq_length = self.seq_length
        h, c = self.init_state(features)
        seq = torch.zeros((batch_size, seq_length), dtype=torch.long)
        seq_log_probs = torch.zeros((batch_size, seq_length))
        for k in range(batch_size):
            tmp_features = features[k:k+1].expand(beam_size)
            tmp_h = h[k: k+1].expand(beam_size)
            tmp_c = c[k: k+1].expand(beam_size)
            done_beams  = self.beam_search(tmp_features, (tmp_h, tmp_c))
        return seq
    def beam_search(self, features, state):
        pass



    def init_state(self, features):
        mean_features = torch.mean(features, dim=1)
        c = self.init_c(mean_features)
        h = self.init_h(mean_features)
        return h,c

class AttCore(nn.Module):
    def __init__(self, opt):
        super(AttCore, self).__init__()
        self.opt = opt
        self.att_feat_size = opt.att_feat_size
        self.embedding_size = opt.embedding_size
        self.hidden_size = opt.hidden_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.proj_features = nn.Linear(self.att_feat_size, self.att_feat_size)
        self.attention = Attention(self.att_feat_size, self.hidden_size)
        #Doubly stochastic attention
        self.selector = Selector(self.hidden_size) if opt.use_selector else None
        self.lstm = nn.LSTMCell(self.att_feat_size + self.embedding_size,
                                self.hidden_size)

        self.dropout = nn.Dropout(p=self.drop_prob_lm)

        self.h_out = nn.Linear(self.hidden_size, self.embedding_size)
        self.ctx_out = nn.Linear(self.att_feat_size, self.embedding_size)
    def forward(self, xt, features, state):
        h, c = state
        projected_features = self.proj_features(features)
        context = self.attention(features, projected_features, h)
        if self.selector:
            context = self.selector(context, h)
        h, c = self.lstm(torch.cat([xt, context], dim=1), (h, c))
        h = self.dropout(h)
        h_out = self.h_out(h)
        ctx_out = self.ctx_out(context)
        final_out = h_out + ctx_out + xt
        return final_out, (h, c)


class Attention(nn.Module):
    def __init__(self, att_feat_size, hidden_size):
        super(Attention, self).__init__()
        self.att_feat_size = att_feat_size
        self.hidden_size = hidden_size
        self.h_att = nn.Linear(hidden_size, att_feat_size)
        self.relu = nn.ReLU(inplace=True)
        self.out_att = nn.Linear(att_feat_size, 1)

    def forward(self, features, features_proj, h_prev):
        att_size = features.size(1)
        h_att = self.h_att(h_prev).unsqueeze(1) + features_proj
        h_att = self.relu(h_att)
        out_att = self.out_att(h_att.reshape(-1, self.att_feat_size))
        out_att = torch.reshape(out_att,(-1, att_size))
        alpha = F.softmax(out_att, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), features).squeeze(1)
        return context


class Selector(nn.Module):
    def __init__(self, hidden_size):
        super(Selector, self).__init__()
        self.f_beta = nn.Linear(hidden_size, 1)

    def forward(self, context, prev_h):
        beta = self.f_beta(prev_h)
        beta = F.sigmoid(beta)
        context = beta*context
        return context
