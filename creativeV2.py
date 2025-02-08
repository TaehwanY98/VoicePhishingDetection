
import spacy
import pandas as pd
# from urllib.request import urlretrieve
from konlpy.tag import Kkma, Komoran, Okt, Mecab # https://konlpy.org/ko/latest/morph/#pos-tagging-with-konlpy 속도분석 Mecab 제일 빠름
# import re
# import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
word_length = 15000

class RNN(nn.Module):
    def __init__(self, input_dims, hidden_dims, layer_dim, batch_size, length,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.GRU =nn.GRU(input_dims, hidden_dims, layer_dim,batch_first=True, bidirectional=False)
        self.Linear1 = nn.Linear(in_features=hidden_dims, out_features=1)
        self.Linear2 = nn.Linear(in_features=length, out_features=100)
        self.Linear3 = nn.Linear(in_features=100, out_features=2)
        self.batchNorm1 = nn.BatchNorm1d(length)
        self.batchNorm2 = nn.BatchNorm1d(100)
        self.Sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.fcl1 = nn.Sequential(self.Linear1, self.relu, self.batchNorm1)
        self.fcl2 = nn.Sequential(self.Linear2, self.relu, self.batchNorm2)
        self.fcl3 = nn.Sequential(self.Linear3, self.Sigmoid)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dims
    def forward(self, x):
        x,_ = self.GRU(x)
        x = self.fcl1(x.squeeze())
        x = self.fcl2(x.squeeze())
        x = self.fcl3(x.squeeze())
        return x
    
def Load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def Save_data(df, file_path):
    df.to_csv(file_path, index=False)
    
def Word_Kkma(text:str):
    kkma= Kkma(max_heap_size=1024)
    pos = kkma.pos(text)
    return pos
def Word_Komoran(text:str):
    komoran= Komoran(max_heap_size=1024)
    pos = komoran.pos(text)
    return pos
def Word_Okt(text:str):
    okt= Okt(max_heap_size=1024)
    pos = okt.pos(text, stem=True, join=True)
    return pos
def Word_Mecab(text:str):
    mecab= Mecab()
    pos = mecab.pos(text)
    return pos 


def temporalAutoLabeling(x, used_tag, nlp=spacy.load("ko_core_news_sm")):
    result = np.ones((word_length,), dtype=np.int32)
    pos = nlp(x)
    dictionary = {token.text:token.pos_  for token in pos if token.pos_ in used_tag}
    dictionary_list = list(dictionary.keys())
    dictionary_value = {word: int(n) for word, n in zip(dictionary_list, range(2, len(dictionary_list)+2))}
    dictionary_index = {token.text: index for index, token in enumerate(pos)}
    for token in pos:
        try :
          result[dictionary_index[token.text]] =  dictionary_value[token.text]
        except KeyError:
          result[dictionary_index[token.text]]= 0
          
    return result

def MIN_MAX_Norm(array):
    return (array-array.min())/(array.max()-array.min())

class Voice_Fishing_Dataset():
    def __init__(self, file_path, mode="sparse", normalization=MIN_MAX_Norm) -> None:
        self.df = Load_data(file_path=file_path)
        print(f"origin length: {len(self.df)}")
        self.mode = mode
        self.normalization = normalization
        self.normal = (self.df['label']==0).sum()
        self.fishing = (self.df['label']==1).sum()
    def __getitem__(self, x):
        voice = self.df["transcript"].loc[x]
        label = self.df["label"].loc[x]
        if self.mode=="komoran":
            used_tag=["NNG", "NNP", "NNB", 'NP', 'NR', 'VV', 'VA', 'VX', 'VCP', 'VCN', 'MM', 'MAG', 'MAJ', 'IC']
            X = temporalAutoLabeling(voice, used_tag=used_tag, nlp=Word_Komoran)
        if self.mode=="okt":
            used_tag=["Noun","Verb", "Adjective"]
            X = temporalAutoLabeling(voice, used_tag=used_tag, nlp=Word_Okt)
        if self.mode=='sparse':
            used_tag=["PROPN","AUX", "VERB","NOUN","NUM"]
            X = temporalAutoLabeling(voice, used_tag=used_tag)
        
        ret = {
            "x": torch.Tensor(X).type(float32),
            "label": torch.Tensor([0, 1]) if int(label)==1 else torch.Tensor([1, 0])
        }

        return ret
    def __len__(self):
        return len(self.df)
    
def train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path, validF):
    net.train()
    history = {}
    for e in range(epoch):
        net.train()
        for sample in tqdm(train_loader, desc="Train: "):

            # print(f"{b+1} batch start")
            X = torch.stack([MIN_MAX_Norm(s["x"].type(torch.int32)) for s in sample], 0).to(DEVICE)
            Y= torch.stack([s["label"] for s in sample], 0).type(torch.int32)
            out = net(X.unsqueeze(-1).type(float32).to(DEVICE))
            
            # print(out.size())
            loss = lossf(out.type(float32).to(DEVICE), Y.type(float32).to(DEVICE))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if valid_loader is not None:
            with torch.no_grad():
                for  key, value in validF(net, valid_loader, e, lossf, DEVICE).items():
                    if e==0:
                        history[key]=[]
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None
    
def valid(net, valid_loader, e, lossf, DEVICE):
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
    far = 0
    for sample in tqdm(valid_loader, desc="Validation: "):
        # print(f"valid {b+1} batch start")
        X = torch.stack([MIN_MAX_Norm(s["x"].type(torch.int32)) for s in sample], 0).to(DEVICE)
        Y= torch.stack([s["label"] for s in sample], 0).type(torch.int32)
        out = net(X.unsqueeze(-1).type(float32).to(DEVICE))
        
        loss += lossf(out.type(float32).to(DEVICE), Y.type(float32).to(DEVICE)).item()
        
        acc += accuracyf(out.squeeze().to(DEVICE), Y.squeeze().to(DEVICE)).item()
        precision += precisionf(out.squeeze().to(DEVICE), Y.squeeze().to(DEVICE)).item()
        f1score += f1scoref(out.squeeze().to(DEVICE), Y.squeeze().to(DEVICE)).item()
        recall += recallf(out.squeeze().to(DEVICE), Y.squeeze().to(DEVICE)).item()
        mar += (1-recallf(out.squeeze().to(DEVICE), Y.squeeze().to(DEVICE))).item()
    
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
    dataset=Voice_Fishing_Dataset("KorCCVi_v2.csv")
    train_set , _ = random_split(dataset, [0.6, 0.4], torch.Generator())
    valid_set = dataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 30
    batch_size = 8
    number_label = [dataset.normal, dataset.fishing]
    lossf = nn.BCEWithLogitsLoss(torch.Tensor([1-x/sum(number_label) for x in(number_label)])).to(DEVICE)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    net = RNN(1, 30, 1, batch_size=batch_size, length=word_length)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    print(net)
    net.to(DEVICE)
    history = train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE=DEVICE, validF=valid, save_path="central")
    history_df = pd.DataFrame(history)
    Save_data(history_df, "./CSV/central.csv")