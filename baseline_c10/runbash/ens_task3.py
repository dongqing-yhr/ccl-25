#!/usr/bin/python3
# -*- coding: utf-8 -*-
# FN+LA 两模型概率平均集成 task3, 解码(弃权+argmax)+框架合法角色门禁, 写 triples 文件。
# 用法: PYTHONPATH=. python3 runbash/ens_task3.py --split A --tau 0.10 --out dataset/A/A_task3_test.json
import argparse, json
import numpy as np, torch, torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from model_task3 import Model

ap=argparse.ArgumentParser()
ap.add_argument('--split',default='dev',choices=['dev','A','B'])
ap.add_argument('--tau',type=float,default=0.10)
ap.add_argument('--out',default=None)
ap.add_argument('--task2_file',default=None,help='override the task2 span file (e.g. a fused old+new union); default = the split slot')
ap.add_argument('--task1_file',default=None,help='override the task1 frame file that feeds the frame-name input and the role whitelist; default = the split slot')
ap.add_argument('--veto_conf',type=float,default=None,help='when set, a span whose top role is outside the frame-legal role list is still emitted if that role probability is >= this value; unset keeps the original hard veto')
A=ap.parse_args()
MC='./chinese-macbert-large/config.json'; MV='./chinese-macbert-large/vocab.txt'
TEST='./dataset/cfn-%s.json'%('dev' if A.split=='dev' else 'test-'+A.split)
sub='dataset/dev' if A.split=='dev' else 'dataset/'+A.split
pre='dev_' if A.split=='dev' else A.split+'_'
T2F=A.task2_file or '%s/%stask2_test.json'%(sub,pre); T1F=A.task1_file or '%s/%stask1_test.json'%(sub,pre)
FI='./dataset/frame_info.json'; FR='./dataset/frame_roles.json'
OUT=A.out or ('%s/%stask3_test.json'%(sub,pre))
print('[ens_task3] split=%s task2=%s task1=%s tau=%s out=%s'%(A.split,T2F,T1F,A.tau,OUT),flush=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self,jf,lf,t2f,t1f,tok,with_frame_name=False):
        self.wfn=with_frame_name; self.tok=tok
        self.all_data=json.load(open(jf,encoding='utf8'))
        self.ori_labels=json.load(open(lf,encoding='utf8'))
        self.sid2frame={}
        if with_frame_name:
            for it in json.load(open(t1f,encoding='utf8')): self.sid2frame[it[0]]=it[1]
        self.idx2label=[]
        for line in self.ori_labels:
            for fes in line['fes']:
                if fes['fe_name'] not in self.idx2label: self.idx2label.append(fes['fe_name'])
        self.dd={}
        for line in self.all_data: self.dd[line['sentence_id']]={'text':line['text'],'target':[line['target'][-1]['start']+1,line['target'][-1]['end']+1]}
        self.data=[]
        for line in json.load(open(t2f,encoding='utf8')):
            rec=self.dd.get(line[0])
            if rec is None: continue
            text=rec['text']; tg=rec['target']; st,en=line[1],line[2]; n=len(text)
            if st<0 or en>=n or st>en: continue
            if en+1<tg[0]: li=[st+1,en+1]
            elif st+1>tg[1]: li=[st+3,en+3]
            else: continue
            if li[1]>=n+4: continue
            self.data.append({'text':text,'label_idx':li,'sentence_id':line[0],'frame':self.sid2frame.get(line[0]),'target':tg,'ori_target':[st,en]})
        self.num_labels=len(self.idx2label)
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        d=self.data[i]; e=self.tok.encode_plus(list(d['text'])); ids=e.data['input_ids']; am=e.data['attention_mask']; t=d['target']
        ids=ids[0:t[0]]+[1]+ids[t[0]:t[1]+1]+[2]+ids[t[1]+1:]; am=am+[1,1]
        if self.wfn and d.get('frame'):
            f=self.tok.convert_tokens_to_ids(list(d['frame'])); ids=ids+[self.tok.sep_token_id]+f; am=am+[1]*(len(f)+1)
        return ids,am,d['label_idx'],d['sentence_id'],d['ori_target']

def collate(data,device=None):
    ml=max(len(x[0]) for x in data); pad=lambda d,v=0:d+[v]*(ml-len(d))
    ids=torch.from_numpy(np.array([pad(x[0]) for x in data],dtype=np.compat.long)).to(device)
    am=torch.from_numpy(np.array([pad(x[1]) for x in data],dtype=np.compat.long)).to(device)
    return ids,am,[x[2] for x in data],[x[3] for x in data],[x[4] for x in data]

dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok=BertTokenizer(vocab_file=MV,do_lower_case=True)
ds_fn=Dataset(TEST,FI,T2F,T1F,tok,True); ds_la=Dataset(TEST,FI,T2F,T1F,tok,False)
assert len(ds_fn)==len(ds_la)
NL=ds_fn.num_labels; idx2label=ds_fn.idx2label; recs=ds_fn.data

def probs(path,ds):
    cfg=BertConfig.from_json_file(MC); cfg.num_labels=NL; cfg.with_null=True
    m=Model(cfg); m.load_state_dict(torch.load(path,map_location='cpu'),strict=False); m=m.to(dev).eval()
    dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=partial(collate,device=dev))
    P=np.zeros((len(ds),NL+1)); pos=0
    with torch.no_grad():
        for ids,am,tg,sid,ori in dl:
            p=F.softmax(m(input_ids=ids,attention_mask=am,target=tg,labels=None,device=dev,for_test=True)['logits'],dim=-1).cpu().numpy()
            P[pos:pos+len(ori)]=p; pos+=len(ori)
    del m; torch.cuda.empty_cache(); return P

Pfn=probs('./saves/model_task3_fn_best.bin',ds_fn); Pla=probs('./saves/model_task3_la_best.bin',ds_la)
avg=(Pfn+Pla)/2.0
allowed={}
fr=json.load(open(FR,encoding='utf8'))
for sid,frame in json.load(open(T1F,encoding='utf8')): allowed.setdefault(sid,set()).update(fr.get(frame,[]))
out=[]; n_null=0; n_veto=0; n_resc=0
for i,r in enumerate(recs):
    if avg[i,NL]>A.tau: n_null+=1; continue
    conf=float(np.max(avg[i,:NL])); name=idx2label[int(np.argmax(avg[i,:NL]))]; sid=r['sentence_id']
    if sid in allowed and name not in allowed[sid]:
        if A.veto_conf is not None and conf>=A.veto_conf: n_resc+=1
        else: n_veto+=1; continue
    out.append([sid,r['ori_target'][0],r['ori_target'][1],name])
json.dump(out,open(OUT,'w',encoding='utf8'),ensure_ascii=False,indent=1)
print('ens: 候选%d 弃权%d 门禁丢%d 保留%d -> %s (veto_conf=%s 越线保留=%d)'%(len(recs),n_null,n_veto,len(out),OUT,A.veto_conf,n_resc))
