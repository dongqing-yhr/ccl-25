import torch
import torch.nn as nn
from transformers import BertModel


class Model(nn.Module):
    """task3 角色打分器: 用"跨度池化 + MLP"替掉现役的逐角色双线性 Q×K 头。

    forward 签名、dataset、loss 里那三个开关(NULL 加权 / logit adjustment / 论元辅助)全都和
    model_task3.Model 一样, 只换读角色这一件事: 现役头给 1010 个角色各配一对 64 维 query/key,
    拿跨度首字和末字的位置做双线性匹配(~132M 参数, 假设"角色由首尾两个字的关系决定");
    这里改成把候选跨度内所有字的隐层做 mean/max/首字/末字四段拼接, 过一个 MLP 直接出 1010 个打分
    (~5M 参数, 假设"角色由整段论元的语义决定")。归纳偏置不同才可能在成员融合里提供互补信息,
    这是现役五成员(同架构五配方)吃不到的那一块。

    帧头那段(L2)只有在 config.num_frames>0 时才创建; 本臂的配方走 fn 的 lambda=0, 所以不创建。
    """

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.num_labels = config.num_labels + 1 if getattr(config, 'with_null', False) else config.num_labels
        self.with_null = getattr(config, 'with_null', False)
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        h = config.hidden_size
        # 跨度表示的四段拼接: 均値 / 最大 / 首字 / 末字
        self.span_norm = nn.LayerNorm(4 * h)
        self.span_dense = nn.Linear(4 * h, h)
        self.span_act = nn.GELU()
        self.span_drop = nn.Dropout(classifier_dropout)
        self.role_head = nn.Linear(h, self.num_labels)

        self.num_frames = getattr(config, 'num_frames', 0) or 0
        self.mtl_frame_weight = float(getattr(config, 'mtl_frame_weight', 0.0) or 0.0)
        if self.num_frames:
            self.frame_head = nn.Linear(h, self.num_frames)

    def pool_span(self, hidden, target):
        """候选跨度的四段池化表示。

        task3 传进来的 target 是候选跨度自己的坐标(label_idx, 已按标记插入移位), 所以这里直接
        hidden[i, lo:hi+1] 就是那一段论元; 与现役头取的首末两字位置完全同一。
        """
        outs = []
        seq_len = hidden.shape[1]
        for i in range(hidden.shape[0]):
            lo = max(0, min(int(target[i][0]), seq_len - 1))
            hi = max(lo, min(int(target[i][1]), seq_len - 1))
            seg = hidden[i, lo:hi + 1]
            outs.append(torch.cat([seg.mean(dim=0), seg.max(dim=0).values, seg[0], seg[-1]], dim=0))
        return torch.stack(outs, dim=0)

    def pool_target_span(self, hidden, target):
        outs = []
        seq_len = hidden.shape[1]
        for i in range(hidden.shape[0]):
            lo = max(0, min(int(target[i][0]), seq_len - 1))
            hi = max(lo + 1, min(int(target[i][1]) + 3, seq_len))
            outs.append(hidden[i, lo:hi].mean(dim=0))
        return torch.stack(outs, dim=0)

    def forward(self, input_ids=None, attention_mask=None, target=None, labels=None, device=None, for_test=False, frame_idx=None,
                gold_frame_idx=None, sentence_id=None):

        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_token = (bert_out["hidden_states"][-4] + bert_out["hidden_states"][-3] + bert_out["hidden_states"][-2] + bert_out["hidden_states"][-1]) / 4
        frame_logits = None
        frame_loss = None
        if self.num_frames:
            frame_logits = self.frame_head(self.pool_target_span(hidden_token, target))

        pooled = self.span_norm(self.pool_span(hidden_token, target))
        token_logits = self.role_head(self.span_drop(self.span_act(self.span_dense(pooled))))

        if for_test:
            loss = None
        else:
            logits_for_loss = token_logits
            tau = getattr(self, 'la_tau', 0.0)
            if tau and getattr(self, 'logit_prior', None) is not None and frame_idx is not None:
                safe_idx = frame_idx.clamp_min(0)
                adj = self.logit_prior[safe_idx] * tau
                valid = (frame_idx >= 0).unsqueeze(1).float()
                logits_for_loss = token_logits + adj * valid
            loss_fc = torch.nn.CrossEntropyLoss()
            null_weight = getattr(self.config, 'null_weight', 0.5)

            if self.with_null:
                normal_mask = (labels < self.num_labels - 1).float()
                null_mask = (labels == self.num_labels - 1).float()

                total_loss = loss_fc(logits_for_loss, labels)

                if null_weight != 1.0:
                    logits_normal = logits_for_loss[:, :-1]
                    logits_null = logits_for_loss[:, -1:]

                    normal_loss = -torch.log_softmax(logits_normal, dim=1)[normal_mask.bool()]
                    null_loss = -torch.log_softmax(logits_null, dim=1)[null_mask.bool()]

                    weighted_loss = normal_loss.mean() + null_weight * null_loss.mean()
                    total_loss = weighted_loss
            else:
                total_loss = loss_fc(logits_for_loss, labels)

            loss = total_loss

            aw = getattr(self.config, 'arghood_weight', 0.0)
            if aw:
                nl_col = token_logits.shape[1] - 1
                some_role = torch.logsumexp(token_logits[:, :nl_col], dim=-1)
                bin_logits = torch.stack([token_logits[:, nl_col], some_role], dim=-1)
                arg_label = (labels < nl_col).long()
                loss = loss + aw * torch.nn.functional.cross_entropy(bin_logits, arg_label)

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
