import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    """task1 帧打分器 = 现役跨度 Q×K 头 + 一个"框架定义"分支。

    加的那一项 = def_weight * <def_query(目标词跨度的平均隐层), def_key(第 j 帧的定义表示)> / sqrt(def_dim)。
    每帧的定义表示由加载好权重的那个编码器自己读"框架名。框架定义句"得到, 只算一次并冻结成 buffer,
    所以分支只多两个 1024x256 的投影 (~0.5M 参数), 定义文本不进训练 batch、不改 dataset。

    def_weight=0 时 def_query/def_key 连创建都不创建, 参数集合与随机数消耗都和 model_task1.Model 逐字节
    一致, 现役四成员的权重照常加载 -> 这一臂相对现役只动了"额外喂一份帧定义先验"这一件事。
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.inner_dim = 64
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, batch_first=True,
                            bidirectional=True)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, config.num_labels * self.inner_dim * 2)

        self.def_weight = float(getattr(config, 'def_weight', 0.0) or 0.0)
        self.def_dim = int(getattr(config, 'def_dim', 256) or 256)
        if self.def_weight:
            self.def_query = nn.Linear(config.hidden_size, self.def_dim)
            self.def_key = nn.Linear(config.hidden_size, self.def_dim, bias=False)
            # 查询侧零初始化: 分支在第 0 步恰好贡献 0, 只靠梯度自己长起来。
            # 不这样做的话 def_weight 就是在赌一个没量过的刻度(两项的 logit 尺度差多少事先不知道),
            # 取大了会在训练早期把现役 Q x K 头的信号淹掉, 取小了整条臂等于没开。
            nn.init.zeros_(self.def_query.weight)
            nn.init.zeros_(self.def_query.bias)
        self.register_buffer('def_emb', torch.zeros(config.num_labels, config.hidden_size))
        self.register_buffer('def_ready', torch.zeros(()))

    @torch.no_grad()
    def build_def_emb(self, tokenizer, texts, batch_size=16, max_length=256):
        """把每帧的"框架名。定义句"压成一个向量, 写进 def_emb 并置 def_ready。

        用当前编码器(即 checkpoint 已加载好的那份)来读定义, 所以定义侧和句子侧在同一个表示空间里;
        池化口径和 forward 读句子时一致(最后四层平均 + 按 attention_mask 取均值)。
        """
        was_training = self.training
        self.eval()
        device = self.def_emb.device
        pooled = []
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            enc = [tokenizer.encode_plus(list(t), add_special_tokens=True, truncation=True,
                                         max_length=max_length) for t in chunk]
            ml = max(len(e.data['input_ids']) for e in enc)
            pad = lambda d, v=0: d + [v] * (ml - len(d))
            input_ids = torch.tensor([pad(e.data['input_ids']) for e in enc], dtype=torch.long,
                                     device=device)
            am = torch.tensor([pad(e.data['attention_mask']) for e in enc], dtype=torch.long,
                              device=device)
            out = self.bert(input_ids=input_ids, attention_mask=am, output_hidden_states=True)
            h = (out["hidden_states"][-4] + out["hidden_states"][-3] + out["hidden_states"][-2]
                 + out["hidden_states"][-1]) / 4
            m = am.unsqueeze(-1).to(h.dtype)
            pooled.append((h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0))
        emb = torch.cat(pooled, dim=0)
        if emb.shape != tuple(self.def_emb.shape):
            raise RuntimeError('def_emb shape %s != buffer shape %s' % (tuple(emb.shape), tuple(self.def_emb.shape)))
        self.def_emb.copy_(emb.to(self.def_emb.dtype))
        self.def_ready.fill_(1.0)
        if was_training:
            self.train()
        return tuple(self.def_emb.shape)

    def pool_span(self, hidden, target):
        """目标词跨度的平均隐层: task1 里 target 就是谓词那一段, 且输入没插标记, 坐标可直接用。"""
        outs = []
        seq_len = hidden.shape[1]
        for i in range(hidden.shape[0]):
            lo = max(0, min(int(target[i][0]), seq_len - 1))
            hi = max(lo, min(int(target[i][1]), seq_len - 1))
            outs.append(hidden[i, lo:hi + 1].mean(dim=0))
        return torch.stack(outs, dim=0)

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim, device):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(device)
        return embeddings

    def forward(self, input_ids=None, attention_mask=None, target=None, labels=None, device=None, for_test=False):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        outputs = self.dense(hidden_token)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        outputs = torch.stack(outputs, dim=-2)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        pos_emb = self.sinusoidal_position_embedding(hidden_token.shape[0], hidden_token.shape[1], 64, device)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)
        logits = logits / self.inner_dim ** 0.5
        token_logits = torch.concat([logits[i][:, target[i][0], target[i][1]].unsqueeze(0) for i in range(len(target))], dim=0)

        if self.def_weight and bool(self.def_ready.item()):
            q = self.def_query(self.pool_span(hidden_token, target))
            k = self.def_key(self.def_emb)
            token_logits = token_logits + self.def_weight * (q @ k.t()) / self.def_dim ** 0.5

        if for_test:
            loss = None
        else:
            loss_fc = torch.nn.CrossEntropyLoss()
            loss = loss_fc(token_logits, labels)
        return {
            "logits": token_logits,
            "loss": loss
        }
