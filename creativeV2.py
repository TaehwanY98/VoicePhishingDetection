from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
# from urllib.request import urlretrieve
# from konlpy.tag import Kkma, Komoran, Okt, Mecab # https://konlpy.org/ko/latest/morph/#pos-tagging-with-konlpy 속도분석 Mecab 제일 빠름
# import re
# import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn,Tensor, stack, float32, argmax, save
# import os
from tqdm import tqdm
import warnings
import random
from utils import parser
from torchmetrics import Accuracy, Precision, Recall, F1Score
import logging
from functools import reduce
word_length = 9000
class SLM(nn.Module):
    def __init__(self, slm, postprocess, batch_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.slm = slm
        self.postprocess = postprocess
        self.Linear1 = nn.Linear(in_features=12, out_features=1)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.fcl1 = nn.Sequential(self.Linear1, self.relu)
        self.batch_size = batch_size
    def forward(self, x):
        x = self.slm(x)
        # x = self.postprocess(x[0])
        x = self.fcl1(x.logits)
        x = self.Sigmoid(x)
        return x
    
def Load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def Save_data(df, file_path):
    df.to_csv(file_path, index=False)
    
class Voice_Phishing_Dataset():
    def __init__(self, file_path, max_length) -> None:
        self.df = Load_data(file_path=file_path)
        print(f"origin length: {len(self.df)}")
        self.normal = [(len(self.df)-len(self.df["label"]==0))/(len(self.df))]
        self.phishing = [(len(self.df)-len(self.df["label"]==1))/(len(self.df))]
        self.max_length = max_length
    def __getitem__(self, x):
        voice = self.df["transcript"].loc[x]
        label = self.df["label"].loc[x]
    
        x = []
        for t in range(0, len(voice), 512):
            try:
                x.append(voice[t:t+512])
            except:
                x.append(voice[t:len(voice)])
        ret = {
            "x": x,
            "y": torch.Tensor([int(label)])
        }
        return ret
    def __len__(self):
        return len(self.df)
    
def postprocessing(out):
    result = torch.zeros((word_length,))
    size = out.size()[0]
    result[:size] = out
    return result    
def chain(*iterables):
    # chain('ABC', 'DEF') --> ['A', 'B', 'C', 'D', 'E', 'F']
    for it in iterables:
        for element in it:
            yield element
    
def train(net, tokenizer, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path, validF):
    net.train()
    history = {}
    for e in range(epoch):
        net.train()
        X=[]
        for sample in tqdm(train_loader, desc="Train: "):
            Y= torch.stack([s["y"] for s in sample], 0).type(torch.int32)
            o_result = []
            for texts in [s['x'] for s in sample]:
                out = []
                X.append(tokenizer(texts,
                          return_tensors="pt", 
                          truncation=True,
                          max_length = 1024,
                          padding=True))
                for text in X:
                    out.append(net(text["input_ids"].to(DEVICE)))
                o_result.append(reduce(lambda x,y: (x.flatten().mean()+y.flatten().mean()).mean(), out).mean())
            loss = lossf(torch.stack(o_result, 0).type(float32).to(DEVICE).unsqueeze(0), Y.type(float32).to(DEVICE))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if valid_loader is not None:
            with torch.no_grad():
                for  key, value in validF(net, tokenizer, valid_loader, e, lossf, DEVICE).items():
                    if e==0:
                        history[key]=[]
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None
    
def valid(net, tokenizer, valid_loader, e, lossf, DEVICE):
    net.eval()
    net.to(DEVICE)
    accuracyf = Accuracy('binary').to(DEVICE)
    precisionf = Precision('binary').to(DEVICE)
    recallf = Recall('binary').to(DEVICE)
    f1scoref = F1Score("binary").to(DEVICE)
    acc=0
    precision=0
    f1score=0
    recall=0
    length = len(valid_loader)
    loss = 0
    mar = 0
    X=[]
    for sample in tqdm(train_loader, desc="Train: "):
        Y= torch.stack([s["y"] for s in sample], 0).type(torch.int32)
        o_result = []
        for texts in [s['x'] for s in sample]:
            out = []
            X.append(tokenizer(texts,
                        return_tensors="pt", 
                        truncation=True,
                        max_length = 1024,
                        padding=True))
            for text in X:
                out.append(net(text["input_ids"].to(DEVICE)))
            o_result.append(reduce(lambda x,y: (x.flatten().mean()+y.flatten().mean()).mean(), out).mean())
        loss = lossf(torch.stack(o_result, 0).type(float32).to(DEVICE).unsqueeze(0), Y.type(float32).to(DEVICE)).item()
        
        acc += accuracyf(torch.stack(o_result, dim=0).unsqueeze(0).to(DEVICE), Y.to(DEVICE)).item()
        precision += precisionf(torch.stack(o_result, dim=0).unsqueeze(0).to(DEVICE), Y.to(DEVICE)).item()
        f1score += f1scoref(torch.stack(o_result, dim=0).unsqueeze(0).to(DEVICE), Y.to(DEVICE)).item()
        recall += recallf(torch.stack(o_result, dim=0).unsqueeze(0).to(DEVICE), Y.to(DEVICE)).item()
        mar += (1-recallf(torch.stack(o_result, dim=0).unsqueeze(0).to(DEVICE), Y.to(DEVICE))).item()
    
    if e is not None:
        print(f"Result epoch {e+1}: loss:{loss/length: .4f} acc:{acc/length: .4f} precision:{precision/length: .4f} f1score:{f1score/length: .4f} recall: {recall/length: .4f}  MAR: {mar/length}")
        
    return {'loss': loss/length, 'acc': acc/length, 'precision': precision/length, 'f1score': f1score/length, "recall": recall/length, "MAR": mar/length} 
 
def seeding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=="__main__":
    logging.basicConfig(filename='logs.log', level=logging.INFO)
    args = parser.FederatedParser()
    seeding(args)
    warnings.filterwarnings('ignore')
    dataset=Voice_Phishing_Dataset("KorCCVi_v2.csv", 512)
    train_set , _ = random_split(dataset, [0.6, 0.4], torch.Generator())
    valid_set = dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 50
    batch_size = 1
    # number_label = [dataset.normal, dataset.phishing]
    lossf = nn.BCEWithLogitsLoss().to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    net = SLM(BertForSequenceClassification.from_pretrained("rmsdud/kobert-classifier").to(DEVICE), postprocessing, batch_size)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(net)
    net.to(DEVICE)
    history = train(net, BertTokenizer.from_pretrained("rmsdud/kobert-classifier"),train_loader, valid_loader, epoch, lossf, optimizer, DEVICE=DEVICE, validF=valid, save_path="central")
    history_df = pd.DataFrame(history)
    Save_data(history_df, "./CSV/central.csv")