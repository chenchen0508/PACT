import shap
#from model import Model
from ATTModel import Model# 
import torch
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import config
from joblib import load


batch_size = 2425
seqLenth = 38
batch_seq_len = []
idx = 0
#num_sample = 10
try_dir = "./MyData/try.jsonl"#"./MyData/datasetAll.jsonl"
try_data_size = 10

"""
def batch2flat(batch_data):
    padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,mask = batch_data
    padded_bifreq = padded_bifreq.view(batch_size,-1)
    padded_trifreq = padded_trifreq.view(batch_size,-1)
    padded_stox = padded_stox.unsqueeze(1)
    padded_ssen = padded_ssen.unsqueeze(1)
    padded_sten = padded_sten.unsqueeze(1)

    padded_dep = [torch.nn.functional.pad(t, (0, seqLenth+1 - t.size(1), 0, seqLenth+1 - t.size(0))) for t in dep]
    dep = torch.stack(padded_dep).view(batch_size,-1)

    label = label.unsqueeze(1)
    data = torch.cat([padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,mask],dim=1)
    global batch_seq_len 
    batch_seq_len = torch.sum(mask,dim=1).numpy()
    data = data.numpy()     #b*[13s+4+(s+1)*(s+1)]
    return data
"""

def batch2flat(batch_data):
    padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,_ = batch_data
    padded_bifreq = padded_bifreq.view(batch_size,-1)
    padded_trifreq = padded_trifreq.view(batch_size,-1)
    padded_stox = padded_stox.unsqueeze(1)
    padded_ssen = padded_ssen.unsqueeze(1)
    padded_sten = padded_sten.unsqueeze(1)

    padded_dep = [torch.nn.functional.pad(t, (0, seqLenth+1 - t.size(1), 0, seqLenth+1 - t.size(0))) for t in dep]
    dep = torch.stack(padded_dep).view(batch_size,-1)

    data = torch.cat([padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep],dim=1)
    label = label.unsqueeze(1)

    data = data.numpy()     #b*[13s+4+(s+1)*(s+1)]
    label = label.numpy()
    return data#,label

def flat2batch(data):
    padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,mask = np.split(data,\
                            [seqLenth,seqLenth*2,seqLenth*3,seqLenth*4,seqLenth*5,seqLenth*6,\
                             seqLenth*7,seqLenth*9,seqLenth*12,\
                                seqLenth*12+1,seqLenth*12+2,seqLenth*12+3,\
                                    seqLenth*12+3+(seqLenth+1)*(seqLenth+1),\
                                        seqLenth*12+3+(seqLenth+1)*(seqLenth+1)+1],axis=1)

    padded_freq = torch.FloatTensor(padded_freq).to(config.device)
    padded_wtox = torch.FloatTensor(padded_wtox).to(config.device)
    padded_pers = torch.LongTensor(padded_pers).to(config.device)
    padded_pos = torch.LongTensor(padded_pos).to(config.device)
    padded_punc = torch.LongTensor(padded_punc).to(config.device)
    padded_wsen = torch.LongTensor(padded_wsen).to(config.device)
    padded_entity = torch.LongTensor(padded_entity).to(config.device)
    padded_bifreq = torch.FloatTensor(padded_bifreq).reshape(-1,seqLenth,2).to(config.device)
    padded_trifreq = torch.FloatTensor(padded_trifreq).reshape(-1,seqLenth,3).to(config.device)
    padded_stox = torch.FloatTensor(padded_stox).squeeze(1).to(config.device)#-
    padded_ssen = torch.LongTensor(padded_ssen).squeeze(1).to(config.device)
    padded_sten = torch.LongTensor(padded_sten).squeeze(1).to(config.device)

    dep = dep.reshape(-1,seqLenth+1,seqLenth+1)
    global idx,batch_seq_len
    dep = [torch.LongTensor(d[:batch_seq_len[idx]+1, :batch_seq_len[idx]+1]) for d in dep]

    label = torch.FloatTensor(label).squeeze(1).to(config.device)

    #mask = np.zeros((try_data_size, seqLenth), dtype=int)
    #mask[:, :batch_seq_len[idx]] = 1
    mask = torch.LongTensor(mask).to(config.device)#b*s

    batch_data = padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,mask
    return batch_data

def predict(batch_data):
    #batch_data = flat2batch(batch_data)
    #model = Model()
    #model.load_state_dict(torch.load('best_model_ATT.pt'))
    #model.eval()
    #with torch.no_grad():
     #   pre ,_ ,_ ,_ = model(batch_data)
    #pre = pre.numpy()
    model = load('Logistic_Regression.joblib')
    pre = model.predict(batch_data)
    return pre

def getOutput(org_map, score): 
    featureI = []
    featureW =[]
    for T in org_map:
        freq = sum(abs(value) for value in T[ : seqLenth])/seqLenth
        wtox = sum(abs(value) for value in T[seqLenth : seqLenth*2])/seqLenth
        pers = sum(abs(value) for value in T[seqLenth*2 : seqLenth*3])/seqLenth
        pos = sum(abs(value) for value in T[seqLenth*3 : seqLenth*4])/seqLenth
        punc = sum(abs(value) for value in T[seqLenth*4 : seqLenth*5])/seqLenth
        wsen = sum(abs(value) for value in T[seqLenth*5 : seqLenth*6])/seqLenth
        entity = sum(abs(value) for value in T[seqLenth*6 : seqLenth*7])/seqLenth
        bifreq = sum(abs(value) for value in T[seqLenth*7 : seqLenth*9])/(2*seqLenth)
        trifreq = sum(abs(value) for value in T[seqLenth*9 : seqLenth*12])/(3*seqLenth)
        stox = sum(abs(value) for value in T[seqLenth*12 : seqLenth*12+1])
        ssen = sum(abs(value) for value in T[seqLenth*12+1 : seqLenth*12+2])
        sten = sum(abs(value) for value in T[seqLenth*12+2: seqLenth*12+3])
        dep = sum(abs(value) for value in T[seqLenth*12+3 : seqLenth*12+3+(seqLenth+1)*(seqLenth+1)])/((seqLenth+1)*(seqLenth+1))
        wsum = np.sum([freq,wtox,pers,pos,punc,wsen,entity,bifreq,trifreq,stox,ssen,sten,dep])
        f_W = freq/wsum,wtox/wsum,pers/wsum,pos/wsum,punc/wsum,wsen/wsum,entity/wsum,bifreq/wsum,trifreq/wsum,stox/wsum,ssen/wsum,sten/wsum,dep/wsum
        featureW.append(f_W)
        sorted_list_with_idx = sorted(enumerate(f_W), key=lambda x: x[1], reverse=True)
        sorted_ind = [index for index, _ in sorted_list_with_idx]
        featureI.append(sorted_ind)

    featureW_ = [sum(col)/len(featureW) for col in zip(*featureW)]
    print("========== Freature weight for Toxicity ==========")
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_list_with_index]
    print(sorted_indices)

    with open('weight.txt','w',encoding='utf-8') as f:
        for w,i in zip(featureW,featureI):
            f.write(str(w))
            f.write('\n')
            f.write(str(i))
            f.write('\n')
        f.write('\n\n\n\n\n\n')
        f.write(str(featureW_))
        f.write('\n')
        f.write(str(sorted_indices))
    
    return featureW_

def getShap():
    try_dataset = MyDataset(try_dir)
    try_loader = DataLoader(dataset=try_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    for _, batch_sample in enumerate(tqdm(try_loader)):
        try_data = batch2flat(batch_sample)
        explainer = shap.KernelExplainer(predict, try_data[:try_data_size])
    
    shap_values = explainer.shap_values(try_data)
    score_pre = predict(try_data)
    score_pre_ = [1 if p > 0.5 else 0 for p in score_pre]
    getOutput(np.array(shap_values),score_pre_)
    return 


import time
start = time.time()
getShap()
end = time.time()
print("Total time : {}min".format((end-start)/60))
