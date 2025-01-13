import torch
import time
from dataLoader import MyDataset,collate_fn
from torch.utils.data import DataLoader
import config
from tqdm import tqdm
from ATTModel import Model
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

    # ========== Calculate Accuracy ========== #
    #score_true_ = [1 if p > 0.5 else 0 for p in score_true]
    #score_pre_ = [1 if p > 0.5 else 0 for p in score_pre]
    #correct = sum([1 if pred == label else 0 for pred, label in zip(score_true_, score_pre_)])
    #accuracy = correct / len(score_pre)
    #print("Accuracy:", accuracy)

    # ========== Calculate Error ========== #
    #with open("predictList.txt",'w') as f:
     #   for a,b in zip(score_true,score_pre):
      #      f.write(str(a)+'\t'+str(b)+'\n')

    # ========== Calculate Weight ========== #
    featureW = []
    featureI = []
    for w in weight_:
        dep = sum(w[24:26])
        sten  = sum(w[22:24])
        ssen = sum(w[20:22])
        stox = sum(w[18:20])
        trifreq = sum(w[16:18])
        bifreq = sum(w[14:16])
        entity = sum(w[12:14])
        wtox = sum(w[10:12])
        freq = sum(w[8:10])
        wsen = sum(w[6:8])
        punc = sum(w[4:6])
        pos = sum(w[2:4])
        pers = sum(w[0:2])
        f_W = [pers,pos,punc,wsen,freq,wtox,entity,bifreq,trifreq,stox,ssen,sten,dep]
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

