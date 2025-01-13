import torch
import time
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from model import Model
import torch.optim as optim
from copy import deepcopy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run():
    train_dataset = MyDataset(config.train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    valid_dataset = MyDataset(config.valid_dir)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    test_dataset = MyDataset(config.test_dir)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    model = Model().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    best_val_loss = 1e18
    best_model = None
    epoch_valid_loss = {}

    for e in range(1,config.epoch + 1):
        # =================== Training =================== #
        model.train()
        train_loss = 0.
        for step, batch_sample in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            _, loss, _ , _= model(batch_sample)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if (step+1) % config.print_step == 0:
                print("Epoch {},Training Loss:{:.4f}".format(e, train_loss / config.print_step))
                train_loss = 0.   

        # =================== Validating =================== #
        model.eval()
        valid_loss = 0.
        with torch.no_grad():           
            step = 0
            for _, batch_sample in enumerate(tqdm(valid_loader)):
                step = step + 1
                _, loss, _, _ = model(batch_sample)
                valid_loss += loss.item()
            valid_loss = valid_loss/step
            epoch_valid_loss[e] = valid_loss
            print("Epoch {}, Valid Loss:{:.4f}".format(e, valid_loss))   
            if valid_loss < best_val_loss:
                print("Epoch {},Get a better model!!!".format(e))
                best_val_loss = valid_loss
                best_model = deepcopy(model)
    torch.save(best_model.state_dict(), 'best_model.pt') #add to save model

    # =================== Testing =================== #
    best_model.eval()
    test_loss = 0.   
    score_true = []
    score_pre = []
    weight_ = []
    with torch.no_grad():
        step = 0
        for _,batch_sample in enumerate(tqdm(test_loader)):
            step = step + 1
            pre ,loss, label, weight = best_model(batch_sample)
            score_true.extend(label.tolist())
            score_pre.extend(pre.tolist())
            test_loss += loss.item()
            weight_.extend(weight.tolist())
        print("Test Loss:{:.4f}".format(test_loss/step))

    # ========== Calculate Weight ========== #
    f_Mask = [0.07593216130541791, 0.09520488049490577, 0.09062566635226992, 0.07259691544582969, 0.05524786454007276, 0.0673218178688034, 0.05797264087928028, 0.07099310825609952, 0.06861479318403946, 0.08189065005794038, 0.08791823086854283, 0.06952683760280289, 0.1061544115311399]
    featureW = []
    featureI = []
    for w in weight_:
        dep = sum(w[78:80])
        sent = sum(w[54:78])
        sten = sent * sum(w[52:54])
        ssen = sent * sum(w[50:52])
        stox = sent * sum(w[48:50])
        phase = sent * sum(w[30:48])
        trifreq = phase * sum(w[28:30])
        bifreq = phase * sum(w[26:28])
        entity = phase * sum(w[24:26])
        word = phase * sum(w[12:24])
        wtox = word * sum(w[10:12])
        freq = word * sum(w[8:10])
        wsen = word * sum(w[6:8])
        punc = word * sum(w[4:6])
        pos = word * sum(w[2:4])
        pers = word * sum(w[0:2])
        f_W = [pers,pos,punc,wsen,freq,wtox,entity,bifreq,trifreq,stox,ssen,sten,dep]
        #change by 20241029
        #f_x = [a + 1 - b  for (a,b) in zip(f_W, f_Mask)]
        #C_sum = sum(f_x)
        #f_W = [c / C_sum for c in f_x]
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

    # ========== Valid Loss for Every Epoch ========== #
    for (k,v) in epoch_valid_loss.items():
        print("epoch: {} ,loss : {}".format(k,v))    


if __name__ == "__main__":
    setup_seed(200)
    start = time.time()
    run()
    end = time.time()
    print("Total time : {}min".format((end-start)/60))

