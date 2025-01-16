#from model import Model
#from singleJob import Model# 
import torch
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import config
from joblib import load


batch_size = 2425
seqLenth = 38
batch_seq_len = []
idx = 0
num_sample = 1000
try_dir = "./MyData/try.jsonl"#"./MyData/datasetAll.jsonl"

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

    mask = np.zeros((num_sample, seqLenth), dtype=int)
    mask[:, :batch_seq_len[idx]] = 1
    mask = torch.LongTensor(mask).to(config.device)#b*s

    batch_data = padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
                padded_entity,padded_bifreq,padded_trifreq,\
                    padded_stox,padded_ssen,padded_sten,\
                        dep,label,mask
    return batch_data

def predict(batch_data):
    #batch_data = flat2batch(batch_data)
    #model = Model()
    #model.load_state_dict(torch.load('best_model_ATT.pt'))#
    #model.eval()
    #with torch.no_grad():
    #    pre ,_ ,_ ,_ = model(batch_data)
    #pre = pre.numpy()

    model = load('Logistic_Regression.joblib')
    pre = model.predict(batch_data)
    return pre

def getOutput(org_map): 
    featureF = []
    featureI = []
    T = sorted(org_map[1], key=lambda x: x[0])
    freq = sum(abs(value) for _, value in T[ : seqLenth])/seqLenth
    wtox = sum(abs(value) for _, value in T[seqLenth : seqLenth*2])/seqLenth
    pers = sum(abs(value) for _, value in T[seqLenth*2 : seqLenth*3])/seqLenth
    pos = sum(abs(value) for _, value in T[seqLenth*3 : seqLenth*4])/seqLenth
    punc = sum(abs(value) for _, value in T[seqLenth*4 : seqLenth*5])/seqLenth
    wsen = sum(abs(value) for _, value in T[seqLenth*5 : seqLenth*6])/seqLenth
    entity = sum(abs(value) for _, value in T[seqLenth*6 : seqLenth*7])/seqLenth
    bifreq = sum(abs(value) for _, value in T[seqLenth*7 : seqLenth*9])/(2*seqLenth)
    trifreq = sum(abs(value) for _, value in T[seqLenth*9 : seqLenth*12])/(3*seqLenth)
    stox = sum(abs(value) for _, value in T[seqLenth*12 : seqLenth*12+1])
    ssen = sum(abs(value) for _, value in T[seqLenth*12+1 : seqLenth*12+2])
    sten = sum(abs(value) for _, value in T[seqLenth*12+2: seqLenth*12+3])
    dep = sum(abs(value) for _, value in T[seqLenth*12+3 : seqLenth*12+3+(seqLenth+1)*(seqLenth+1)])/((seqLenth+1)*(seqLenth+1))
    wsum = np.sum([freq,wtox,pers,pos,punc,wsen,entity,bifreq,trifreq,stox,ssen,sten,dep])
    featureW_ = freq/wsum,wtox/wsum,pers/wsum,pos/wsum,punc/wsum,wsen/wsum,entity/wsum,bifreq/wsum,trifreq/wsum,stox/wsum,ssen/wsum,sten/wsum,dep/wsum
    print("========== Freature weight for Toxicity ==========")
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_list_with_index]
    print(sorted_indices)

    NT = sorted(org_map[0], key=lambda x: x[0])
    freq = sum(abs(value) for _, value in NT[ : seqLenth])/seqLenth
    wtox = sum(abs(value) for _, value in NT[seqLenth : seqLenth*2])/seqLenth
    pers = sum(abs(value) for _, value in NT[seqLenth*2 : seqLenth*3])/seqLenth
    pos = sum(abs(value) for _, value in NT[seqLenth*3 : seqLenth*4])/seqLenth
    punc = sum(abs(value) for _, value in NT[seqLenth*4 : seqLenth*5])/seqLenth
    wsen = sum(abs(value) for _, value in NT[seqLenth*5 : seqLenth*6])/seqLenth
    entity = sum(abs(value) for _, value in NT[seqLenth*6 : seqLenth*7])/seqLenth
    bifreq = sum(abs(value) for _, value in NT[seqLenth*7 : seqLenth*9])/(2*seqLenth)
    trifreq = sum(abs(value) for _, value in NT[seqLenth*9 : seqLenth*12])/(3*seqLenth)
    stox = sum(abs(value) for _, value in NT[seqLenth*12 : seqLenth*12+1])
    ssen = sum(abs(value) for _, value in NT[seqLenth*12+1 : seqLenth*12+2])
    sten = sum(abs(value) for _, value in NT[seqLenth*12+2: seqLenth*12+3])
    dep = sum(abs(value) for _, value in NT[seqLenth*12+3 : seqLenth*12+3+(seqLenth+1)*(seqLenth+1)])/((seqLenth+1)*(seqLenth+1))
    wsum = np.sum([freq,wtox,pers,pos,punc,wsen,entity,bifreq,trifreq,stox,ssen,sten,dep])
    featureW_ = freq/wsum,wtox/wsum,pers/wsum,pos/wsum,punc/wsum,wsen/wsum,entity/wsum,bifreq/wsum,trifreq/wsum,stox/wsum,ssen/wsum,sten/wsum,dep/wsum
    print("========== Freature weight for Non Toxicity ==========")
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, value in sorted_list_with_index]
    print(sorted_indices)
    featureF.append(featureW_)
    featureI.append(sorted_indices)
    with open('weight.txt','a+',encoding='utf-8') as f:
        for w,i in zip(featureF,featureI):
            f.write(str(w))
            f.write('\n')
            f.write(str(i))
            f.write('\n')
    return featureW_

def getLime():
    try_dataset = MyDataset(try_dir)
    try_loader = DataLoader(dataset=try_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    for _, batch_sample in enumerate(tqdm(try_loader)):
        try_data = batch2flat(batch_sample)
        explainer = LimeTabularExplainer(training_data = try_data, mode='regression', discretize_continuous=True)
    
    weight = []
    global idx
    for idx in range(batch_size):
        print(idx)
        sample_data = try_data[idx]
        exp = explainer.explain_instance(
            data_row = sample_data,
            predict_fn = predict,
            num_samples=num_sample,
            num_features= 13*seqLenth+4+(seqLenth+1)*(seqLenth+1)  
        )
        weight.append(getOutput(exp.as_map()))
    print("================== Final ==================")
    featureW_ = np.mean(weight, axis=0)
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_list_with_index]
    print(sorted_indices)
    return 

import time
start = time.time()
getLime()
end = time.time()
print("Total time : {}min".format((end-start)/60))

