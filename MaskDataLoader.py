from torch.utils.data import Dataset
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import config


class MyDataset(Dataset):
    def __init__(self, fileDir):
       dataLine = []
       with open(fileDir,'r',encoding='utf-8') as f:
           for line in f:
               dataLine.append(json.loads(line))
       self.data = dataLine 

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def collate_fn(data):
    freq = []
    wtox = []
    pers = []
    pos = []
    punc = []
    wsen = []
    entity = []
    bifreq = []
    trifreq = []
    stox = []
    ssen = []
    sten = []
    dep = []
    label = []
    for line in data:
        freq.append(torch.tensor(line['freq']))
        wtox.append(torch.tensor(line['toxicity']))
        pers.append(torch.tensor(line['person']))
        pos.append(torch.tensor(line['pos']))
        punc.append(torch.tensor(line['punc']))
        wsen.append(torch.LongTensor(line['sentiment']))

        entity.append(torch.tensor(line['entity']))
        bifreq.append(torch.tensor(line['Bifreq']))
        trifreq.append(torch.tensor(line['Trifreq']))

        stox.append(line['toxicityS'])
        ssen.append(line['sentimentS'])
        sten.append(line['tense'])

        #dep.append(torch.tensor(line['dep']))
        dep.append(torch.zeros_like(torch.tensor(line['dep'])))
        label.append(line['score'])

    padded_freq = pad_sequence(freq, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_wtox = pad_sequence(wtox, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_pers = pad_sequence(pers, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_pos = pad_sequence(pos, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_punc = pad_sequence(punc, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_wsen = pad_sequence(wsen, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    
    padded_entity = pad_sequence(entity, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth
    padded_bifreq = pad_sequence(bifreq, batch_first=True, padding_value = 0) #batch_size * max_seq_lenth * 2
    padded_trifreq = pad_sequence(trifreq, batch_first=True, padding_value = 0)#batch_size * max_seq_lenth * 3

    padded_stox = torch.FloatTensor(stox)#batch_size
    padded_ssen = torch.LongTensor(ssen)#batch_size
    padded_sten = torch.tensor(sten)#batch_size

    label  = torch.FloatTensor(label)#batch_size
    mask = torch.tensor(padded_freq!=0)#batch_size * max_seq_lenth

    padded_freq = padded_freq.to(config.device)
    padded_wtox = padded_wtox.to(config.device)
    padded_pers = padded_pers.to(config.device)
    padded_pos = padded_pos.to(config.device)
    padded_punc = padded_punc.to(config.device)
    padded_wsen = padded_wsen.to(config.device)
    padded_entity = padded_entity.to(config.device)
    padded_bifreq = padded_bifreq.to(config.device)
    padded_trifreq = padded_trifreq.to(config.device)
    padded_stox = padded_stox.to(config.device)
    padded_ssen = padded_ssen.to(config.device)
    padded_sten = padded_sten.to(config.device)
    label = label.to(config.device)
    mask = mask.to(config.device)
    
    padded_freq = torch.zeros_like(padded_freq)#0
    padded_wtox = torch.zeros_like(padded_wtox)#1
    padded_pers = torch.zeros_like(padded_pers)#2
    padded_pos = torch.zeros_like(padded_pos)#3
    padded_punc = torch.zeros_like(padded_punc)#4
    padded_wsen = torch.zeros_like(padded_wsen)#5
    padded_entity = torch.zeros_like(padded_entity)#6
    padded_bifreq = torch.zeros_like(padded_bifreq)#7
    padded_trifreq = torch.zeros_like(padded_trifreq)#8
    #padded_stox = torch.zeros_like(padded_stox)#9
    padded_ssen = torch.zeros_like(padded_ssen)#10
    padded_sten = torch.zeros_like(padded_sten)#11
    #dep = torch.zeros_like(dep)#12
    
    return padded_freq,padded_wtox,padded_pers,padded_pos,padded_punc,padded_wsen,\
            padded_entity,padded_bifreq,padded_trifreq,\
                padded_stox,padded_ssen,padded_sten,\
                    dep,label,mask 
    
