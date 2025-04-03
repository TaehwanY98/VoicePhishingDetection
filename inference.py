from transformers import BertForSequenceClassification, BertTokenizer
import pandas as pd
# from konlpy.tag import Kkma, Komoran, Okt, Mecab # https://konlpy.org/ko/latest/morph/#pos-tagging-with-konlpy 속도분석 Mecab 제일 빠름
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import nn,float32
from tqdm import tqdm
import warnings
import random
from functools import reduce
# word_length = 9000
batch_size =1

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

# def postprocessing(out):
#     result = torch.zeros((word_length,))
#     size = out.size()[0]
#     result[:size] = out
#     return result
    
def infer(net, valid_loader, tokenizer,DEVICE):
    net.eval()
    net.to(DEVICE)
    X=[]
    for sample in tqdm(valid_loader, desc="infer: "):
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
        print(o_result)
    # return client_result
         
def seeding(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__=="__main__":
    warnings.filterwarnings('ignore')
    client_id = 77
    script = input("scripts:")
    try:
        client_scripts = pd.read_csv("./Client/client1.csv")
    except:
        client_scripts = pd.DataFrame(columns=["id", "transcript"])
    new_script = pd.DataFrame( data={"id":[client_id], "transcript": [str(script)]})
    pd.concat([client_scripts, new_script], keys=["id", "transcript"]).to_csv("./Client/client1.csv", index=False)
    
    infer_set = Voice_Phishing_Dataset("./Client/client1.csv")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_loader = DataLoader(infer_set, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    net = SLM(BertForSequenceClassification.from_pretrained("rmsdud/kobert-classifier"), postprocessing, batch_size)
    net.to(DEVICE)
    tokenizer = BertTokenizer.from_pretrained("rmsdud/kobert-classifier")
    # print("Voice Phishing", infer(net, infer_loader, tokenizer, DEVICE))
    infer(net, infer_loader, tokenizer, DEVICE)
    
    