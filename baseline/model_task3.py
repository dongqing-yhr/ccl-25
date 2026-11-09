import os
import torch
import torch.nn as nn
from transformers import BertModel

# 角色头读哪几层隐状态: 现役五成员 = 最后四层平均(avg4)。last = 只取最后一层。
# 由 train_task3.py / predict_task3.py 从 --layer_fusion 写入(不传则维持 avg4)。
LAYER_FUSION = os.environ.get('LAYER_FUSION', 'avg4')
# GlobalPointer 打分头的 inner_dim(同时是正弦位置编码的维数)。--inner_dim 写入, 默认 64 = 现役。
INNER_DIM = int(os.environ.get('INNER_DIM', '64'))
# 负样本(NULL 类跨度)在角色交叉熵里的倍率。对应对手任务三的 neg_boost 那一维。
# 1.0 = 关闭, 走原来的 CrossEntropyLoss 代码路径, 与现役逐字节一致。由 train_task3.py 从 --neg_boost 写入。
NEG_BOOST = float(os.environ.get('NEG_BOOST', '1.0'))
# RoPE 方案: 'ours'(默认, 2k/d 频率, 现役成员训练用) 或 'theirs'(k/d 频率, 对方 LERT 权重训练用)。
# 两种方案的 dense 权重不兼容, 必须与训练时一致。
ROPE_SCHEME = os.environ.get('ROPE_SCHEME', 'ours')


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        # NULL 弃权机制：增加 NULL 类
        self.num_labels = config.num_labels + 1 if getattr(config, 'with_null', False) else config.num_labels
        self.inner_dim = INNER_DIM
        self.bert = BertModel(config)
        # LSTM 仅在配置要求时创建(现役成员默认创建但 forward 中不使用; 对方 LERT 权重不含 LSTM 参数,
        # 必须设 use_lstm=False 才能干净装载)。
        if getattr(config, 'use_lstm', True):
            self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=1, batch_first=True,
                                bidirectional=True)
        else:
            self.lstm = None
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.dense = nn.Linear(config.hidden_size, self.num_labels * self.inner_dim * 2)
        self.with_null = getattr(config, 'with_null', False)

        # L2 共享编码器多任务: 帧头只有在 config.num_frames>0 (即 λ>0) 时才创建,
        # 默认关 -> 现役五成员的参数名/形状/前向路径逐字节不变, 老权重照常加载。
        self.num_frames = getattr(config, 'num_frames', 0) or 0
        self.mtl_frame_weight = float(getattr(config, 'mtl_frame_weight', 0.0) or 0.0)
        if self.num_frames:
            self.frame_head = nn.Linear(config.hidden_size, self.num_frames)

    def pool_target_span(self, hidden, target):
        """Mean hidden over the target word: the two marker tokens and the characters between them.

        A frame is evoked by the target word, not by the whole sentence or by one candidate span,
        so the frame head reads exactly the region the role head already indexes its span at.
        """
        outs = []
        seq_len = hidden.shape[1]
        for i in range(hidden.shape[0]):
            lo = max(0, min(int(target[i][0]), seq_len - 1))
            hi = max(lo + 1, min(int(target[i][1]) + 3, seq_len))
            outs.append(hidden[i, lo:hi].mean(dim=0))
        return torch.stack(outs, dim=0)

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

    def sinusoidal_position_embedding_theirs(self, batch_size, seq_len, output_dim, device):
        """对方 t3model 的 RoPE: 频率 k/d(我们的是 2k/d)。dense 权重在对方频率下训练, 混用会归零。
        返回形状 (1, seq_len, 1, output_dim) 用于广播到 (batch, seq_len, num_labels, inner_dim)。"""
        position_ids = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(-1)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000, -2.0 * indices / output_dim)
        ang = position_ids * indices.unsqueeze(0)
        cos = torch.cos(ang).repeat_interleave(2, dim=-1)
        sin = torch.sin(ang).repeat_interleave(2, dim=-1)
        # 形状: (seq_len, output_dim) -> (1, seq_len, 1, output_dim)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        return cos, sin

    def forward(self, input_ids=None, attention_mask=None, target=None, labels=None, device=None, for_test=False, frame_idx=None,
                gold_frame_idx=None, sentence_id=None):

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
        # hidden_token = bert_out.last_hidden_state
        frame_logits = None
        frame_loss = None
        if self.num_frames:
            frame_logits = self.frame_head(self.pool_target_span(hidden_token, target))
        outputs = self.dense(hidden_token)
        # lstm_out, (hidden, _) = self.lstm(hidden_token)
        # outputs = self.dense(lstm_out)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        # pos_emb:(batch_size, seq_len, inner_dim)
        if ROPE_SCHEME == 'theirs':
            cos_pos, sin_pos = self.sinusoidal_position_embedding_theirs(
                hidden_token.shape[0], hidden_token.shape[1], self.inner_dim, device)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos
        else:
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
        token_logits = torch.concat([logits[i][:, target[i][0], target[i][1]].unsqueeze(0) for i in range(len(target))], dim=0)

        if for_test:
            loss = None
        else:
            # logit adjustment: 训练期把 tau*log(prior[该样本帧]) 加进角色打分再算 CE; 推理用原始 token_logits
            logits_for_loss = token_logits
            tau = getattr(self, 'la_tau', 0.0)
            if tau and getattr(self, 'logit_prior', None) is not None and frame_idx is not None:
                safe_idx = frame_idx.clamp_min(0)
                adj = self.logit_prior[safe_idx] * tau
                valid = (frame_idx >= 0).unsqueeze(1).float()
                logits_for_loss = token_logits + adj * valid
            loss_fc = torch.nn.CrossEntropyLoss()
            # NULL 损失权重：默认 0.5，可通过 self.config.null_weight 调整
            null_weight = getattr(self.config, 'null_weight', 0.5)

            if NEG_BOOST != 1.0:
                # 对手任务三的 neg_boost 类比: 每个候选跨度的交叉熵照旧算(softmax 仍含 NULL 列),
                # 只把标签为 NULL 的那批乘上倍率, 分母仍是这一批的跨度数, 所以正样本项不变、只放大负样本项。
                # 与 model_task2.py 的 POS_WEIGHT 同一风格, 不是 BCE 的 pos_weight, 也不是下面 null_weight 那条支路。
                ce_all = torch.nn.functional.cross_entropy(logits_for_loss, labels, reduction='none')
                is_null = (labels == self.num_labels - 1).float()
                total_loss = (ce_all * (1.0 + (NEG_BOOST - 1.0) * is_null)).sum() / labels.shape[0]
            elif self.with_null:
                # 分离 NULL 类和其他类的损失
                # labels: 0~num_labels-1 为正常类，num_labels 为 NULL 类
                normal_mask = (labels < self.num_labels - 1).float()
                null_mask = (labels == self.num_labels - 1).float()

                # 计算总损失
                total_loss = loss_fc(logits_for_loss, labels)

                # 如果配置了 NULL 权重，调整 NULL 样本的损失贡献
                if null_weight != 1.0:
                    # 重新计算加权损失
                    logits_normal = logits_for_loss[:, :-1]  # 去掉 NULL 类
                    logits_null = logits_for_loss[:, -1:]    # NULL 类

                    # 正常样本的负对数似然
                    normal_loss = -torch.log_softmax(logits_normal, dim=1)[normal_mask.bool()]
                    null_loss = -torch.log_softmax(logits_null, dim=1)[null_mask.bool()]

                    # 加权
                    weighted_loss = normal_loss.mean() + null_weight * null_loss.mean()
                    total_loss = weighted_loss
            else:
                total_loss = loss_fc(logits_for_loss, labels)

            loss = total_loss

            # 辅助任务"是不是论元": 把角色列的证据合成一列, 与最后一列(NULL)做二分类。
            # 两种约定都把 NULL 放在最后一列, 所以这里按列数取, 不依赖 self.with_null。
            aw = getattr(self.config, 'arghood_weight', 0.0)
            if aw:
                nl_col = token_logits.shape[1] - 1
                some_role = torch.logsumexp(token_logits[:, :nl_col], dim=-1)
                bin_logits = torch.stack([token_logits[:, nl_col], some_role], dim=-1)
                arg_label = (labels < nl_col).long()
                loss = loss + aw * torch.nn.functional.cross_entropy(bin_logits, arg_label)

            # 帧损失(L2): 同一个 batch 里一句可能有多个候选跨度, 不去重会让长句的帧梯度
            # 按跨度数成倍放大, 所以每个 sentence_id 只取第一条带金标帧的行。
            fw = self.mtl_frame_weight
            frame_loss = None
            if fw and frame_logits is not None and gold_frame_idx is not None:
                gf = gold_frame_idx.tolist()
                keep, seen = [], set()
                for i in range(len(gf)):
                    if gf[i] < 0:
                        continue
                    sid = sentence_id[i] if sentence_id is not None else i
                    if sid in seen:
                        continue
                    seen.add(sid)
                    keep.append(i)
                if keep:
                    idx = torch.as_tensor(keep, device=frame_logits.device, dtype=torch.long)
                    frame_loss = torch.nn.functional.cross_entropy(frame_logits[idx], gold_frame_idx[idx])
                    loss = loss + fw * frame_loss

        return {
            "logits": token_logits,
            "loss": loss,
            "frame_logits": frame_logits,
            "frame_loss": frame_loss
        }

