import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import torch
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score
from joblib import dump


batch_size = 2425
seqLenth = 38#整个文件是一个batch，这个batch中最长的句子的长度
try_dir = "./MyData/try.jsonl"#"./MyData/datasetAll.jsonl"

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
    return data,label

def getOutput(org_map): 
    T = org_map
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
    featureW_ = freq/wsum,wtox/wsum,pers/wsum,pos/wsum,punc/wsum,wsen/wsum,entity/wsum,bifreq/wsum,trifreq/wsum,stox/wsum,ssen/wsum,sten/wsum,dep/wsum
    print("========== Freature weight for Toxicity ==========")
    print(featureW_)
    sorted_list_with_index = sorted(enumerate(featureW_), key=lambda x: x[1], reverse=True)
    sorted_indices = [index for index, _ in sorted_list_with_index]
    print(sorted_indices)
    return 

def getDTree():
    try_dataset = MyDataset(try_dir)
    try_loader = DataLoader(dataset=try_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for _, batch_sample in enumerate(tqdm(try_loader)):
        try_data,label = batch2flat(batch_sample)

    X = pd.DataFrame(try_data)#, columns=feature_names)
    y = label

    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X, y)

    feature_importance = dt_regressor.feature_importances_
    getOutput(feature_importance)

def getMSELoss():
    train_dir = './MyData/train.jsonl'
    train_dataset = MyDataset(train_dir)
    global batch_size
    batch_size = 1940
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for _, batch_sample in enumerate(tqdm(train_loader)):
        train_data,train_label = batch2flat(batch_sample)

    model = DecisionTreeRegressor(random_state=42)
    model.fit(train_data, train_label)
    dump(model, 'Decision_Tree.joblib')

    test_dir = './MyData/test.jsonl'
    test_dataset = MyDataset(test_dir)
    batch_size = 244#给test在最后添加了一条数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for _, batch_sample in enumerate(tqdm(test_loader)):
        test_data,test_label = batch2flat(batch_sample)
    test_pre = model.predict(test_data)
    mse = mean_squared_error(test_label, test_pre)
    r2 = r2_score(test_label, test_pre)
    print("Test Loss:{:.4f}".format(mse))
    return

#getDTree()
getMSELoss()





