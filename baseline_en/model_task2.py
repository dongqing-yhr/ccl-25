import os
import torch
import torch.nn as nn
from transformers import BertModel

# 嵌套对损失权重: 被别的论元严格包住的正样本, 损失上权重。1.0 = 关闭(基线行为)。
NESTED_WEIGHT = float(os.environ.get('NESTED_WEIGHT', '1.0'))
# 目标词感知: 1 = 把目标词区间的隐状态均值加到每个 token 表示上, 0 = 关闭(基线行为)。
TARGET_FUSE = int(os.environ.get('TARGET_FUSE', '0'))
# 正样本损失权重(对应对手 BCE 的 pos_weight 那一维; 我们用的是 GlobalPointer 不是 BCE, 只能类比不能等价)。
# 1.0 = 关闭, 与现役逐字节一致。由 train_task2.py 从 --pos_weight 写入。
POS_WEIGHT = float(os.environ.get('POS_WEIGHT', '1.0'))
# 跨度头读哪几层隐状态: last / avg2 / avg4。由 train_task2.py、predict_task2.py 从 --layer_fusion 写入。
LAYER_FUSION = os.environ.get('LAYER_FUSION', 'last')
# GlobalPointer 头的 inner_dim(同时是正弦位置编码的维数)。由 --inner_dim 写入, 默认 64 = 现役。
INNER_DIM = int(os.environ.get('INNER_DIM', '64'))


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = INNER_DIM
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(768, 768 // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.num_labels * self.inner_dim * 2)


    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1]*len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, labels=None, target=None, device=None, for_test=False):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        if LAYER_FUSION == 'last':
            hidden_token = bert_out.last_hidden_state
        else:
            hs = bert_out.hidden_states
            if LAYER_FUSION == 'avg4':
                hidden_token = (hs[-4] + hs[-3] + hs[-2] + hs[-1]) / 4
            elif LAYER_FUSION == 'avg2':
                hidden_token = (hs[-2] + hs[-1]) / 2
            else:
                raise ValueError('unknown LAYER_FUSION: %s' % LAYER_FUSION)
        # hidden_token = (bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 2
        if TARGET_FUSE and target is not None:
            # 编码序列中目标词区间 = [t0+1, t1+1] (输入在 t0 和 t1+2 处插了标记 1/2)
            bs, L, _ = hidden_token.shape
            t = torch.as_tensor(target, device=hidden_token.device)
            ts = (t[:, 0] + 1).clamp(0, L - 1)
            te = (t[:, 1] + 1).clamp(0, L - 1)
            pos = torch.arange(L, device=hidden_token.device).unsqueeze(0)
            tmask = ((pos >= ts.unsqueeze(1)) & (pos <= te.unsqueeze(1))).unsqueeze(-1).float()
            h_target = (hidden_token * tmask).sum(1) / tmask.sum(1).clamp(min=1.0)
            hidden_token = hidden_token + h_target.unsqueeze(1)
        outputs = self.dense(hidden_token)
        # lstm_out, (hidden, _) = self.lstm(hidden_token)
        # outputs = self.dense(lstm_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim)
        pos_emb = self.sinusoidal_position_embedding(hidden_token.shape[0], hidden_token.shape[1], self.inner_dim, device)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.inner_dim ** 0.5
        logits = logits.squeeze(1)

        if for_test:
            loss = None
        else:
            # loss_fct = nn.CrossEntropyLoss()
            # loss = loss_fct(H_logits.view(-1, 2), labels.view(-1))
            loss = self.compute_loss(logits, labels, attention_mask)
        return {
            "logits": logits,
            "loss": loss
        }

    def nested_pair_mask(self, labels):
        """labels: [b, L, L]。返回同形状, 被别的正样本严格包住的位置为 1。"""
        cs = torch.flip(torch.cummax(torch.flip(labels, dims=[2]), dim=2).values, dims=[2])
        cs_a = torch.zeros_like(cs)
        cs_a[:, 1:, :] = cs[:, :-1, :]
        term_a = torch.cummax(cs_a, dim=1).values
        rs = torch.zeros_like(cs)
        rs[:, :, :-1] = cs[:, :, 1:]
        term_b = torch.cummax(rs, dim=1).values
        return ((term_a > 0) | (term_b > 0)).float()

    def compute_loss(self, logits, labels, attention_mask):
        H_attention_mask = torch.triu(
            torch.matmul(attention_mask.unsqueeze(2).float(), attention_mask.unsqueeze(1).float()), diagonal=0)
        if NESTED_WEIGHT != 1.0:
            pos_w = labels + (NESTED_WEIGHT - 1.0) * self.nested_pair_mask(labels) * labels
        else:
            pos_w = labels
        if POS_WEIGHT != 1.0:
            pos_w = pos_w * POS_WEIGHT
        loss1 = torch.sum(torch.exp(-logits) * H_attention_mask * pos_w, dim=(1, 2))
        loss2 = torch.sum(torch.exp(logits) * H_attention_mask * (1 - labels), dim=(1, 2))
        loss = torch.sum(torch.log(1 + loss1 + 1e-9) + torch.log(1 + loss2 + 1e-9)) / (H_attention_mask.shape[0])
        return loss
