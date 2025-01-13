import torch
import torch.nn as nn
import config
import math
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_pers = nn.Embedding(config.pers,config.pers_emb)
        self.emb_pos = nn.Embedding(config.pos,config.pos_emb)
        self.emb_punc = nn.Embedding(config.punc,config.punc_emb)
        self.emb_wsen = nn.Embedding(config.wsen,config.wsen_emb)
        self.line_pers = nn.Linear(config.pers_emb,2)
        self.line_pos = nn.Linear(config.pos_emb,2)
        self.line_punc = nn.Linear(config.punc_emb,2)
        self.line_wsen = nn.Linear(config.wsen_emb,2)
        self.line_freq = nn.Linear(1,2)
        self.line_wtox = nn.Linear(1,2)

        self.wordAttention = MultiHeadedAttention(config.word_hidden,config.head_num,config.dropout)
        self.wordLinear = nn.Linear(config.word_hidden,config.word_final)

        self.emb_entity = nn.Embedding(config.entity,config.entity_emb)
        self.line_entity = nn.Linear(config.entity_emb,2)
        self.line_bifreq = nn.Linear(2,2)
        self.line_trifreq = nn.Linear(3,2)

        self.phaseAttention = MultiHeadedAttention(config.phase_hidden,config.head_num,config.dropout)
        self.phaseLinear = nn.Linear(config.phase_hidden,config.phase_final)
        self.phasePool = nn.AdaptiveAvgPool1d(1)

        self.emb_ssen = nn.Embedding(config.ssen,config.ssen_emb)
        self.emb_sten = nn.Embedding(config.sten,config.sten_emb)
        self.line_ssen = nn.Linear(config.ssen_emb,2)
        self.line_sten = nn.Linear(config.sten_emb,2)
        self.line_stox = nn.Linear(1,2)

        self.sentAttention = MultiHeadedAttention(config.sent_hidden,config.head_num,config.dropout)
        self.sentLinear = nn.Linear(config.sent_hidden,config.sent_final)
        
        self.depCNN = MyCNN(config.dep, config.dep_emb, config.num_filter)
        self.depLinear = nn.Linear(config.num_filter, config.dep_final)
        self.depAttention = MultiHeadedAttention(config.sent_final+config.dep_final,config.head_num,config.dropout)

        self.fcn = nn.Linear(config.sent_final+config.dep_final,1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

    def forward(self, batch_sent):
        padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
            padded_entity,padded_bifreq,padded_trifreq,\
                padded_stox,padded_ssen,padded_sten,\
                    dep,label,mask = batch_sent
        
        emb_pers = self.emb_pers(padded_pers)
        emb_pos = self.emb_pos(padded_pos)
        emb_punc = self.emb_punc(padded_punc)
        emb_wsen = self.emb_wsen(padded_wsen)
        emb_freq = padded_freq.unsqueeze(-1)
        emb_wtox = padded_wtox.unsqueeze(-1)
        line_pers = self.line_pers(emb_pers)
        line_pos = self.line_pos(emb_pos)
        line_punc = self.line_punc(emb_punc)
        line_wsen  = self.line_wsen(emb_wsen)
        line_freq = self.line_freq(emb_freq)
        line_wtox = self.line_wtox(emb_wtox)

        
        score = self.sigmoid(s)
        loss = self.loss(score, label)
        return score, loss, label, torch.cat([weight_w,weight_p,weight_s,weight_d],dim=1)

 
class MyCNN(nn.Module):
    def __init__(self, num_dep,emb_size,num_filter):
        super(MyCNN,self).__init__()
        self.emb = nn.Embedding(num_dep,emb_size)
        self.conv = nn.Conv2d(in_channels=emb_size, out_channels=num_filter, kernel_size=3, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        
    def forward(self, dep):
        emb = self.emb(dep)
        emb = emb.permute(2,0,1)
        conv_out = F.relu(self.conv(emb))
        pooled_out = self.pooling(conv_out).squeeze(-1).squeeze(-1)
        return pooled_out


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, hidden_size, heads_num, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        self.per_head_size = hidden_size // heads_num

        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, key, value, query, mask):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size  x seq_length]
            mask is 0 if it is masked

        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, hidden_size = key.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                contiguous(). \
                view(batch_size, seq_length, heads_num, per_head_size). \
                transpose(1, 2)

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, hidden_size)

        query, key, value = [l(x).view(batch_size, -1, heads_num, per_head_size).transpose(1,2) for l, x in zip(self.linear_layers, (query, key, value))]

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(float(per_head_size))
        mask = mask. \
            unsqueeze(1). \
            repeat(1, seq_length, 1). \
            unsqueeze(1)
        mask = mask.float()
        mask = (1.0 - mask) * -10000.0
        scores = scores + mask
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)

        #weighted_sum = torch.einsum('bsl,bsh->bh', probs.squeeze(1), value.squeeze(1))  # [batch_size, hidden_size]
        #feature_weights = F.softmax(weighted_sum, dim=-1)  # [batch_size, hidden_size]
        weighted_sum = output.mean(dim=1)
        feature_weights = F.softmax(weighted_sum, dim=-1)  # [batch_size, hidden_size]
        #========changed at 20240929========#
        return output, feature_weights

    