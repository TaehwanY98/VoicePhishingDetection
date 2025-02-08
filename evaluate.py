from creativeV2 import *
from torch import nn
from torch.utils.data import DataLoader
import torch
args = parser.FederatedParser()
seeding(args)
if __name__=="__main__":
    warnings.filterwarnings('ignore')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size= 8
    net = RNN(1, 30, 1, batch_size=batch_size, length=word_length)
    testset = Voice_Fishing_Dataset("KorCCVi_v2.csv")
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)
    number_label = [testset.normal, testset.fishing]
    net.load_state_dict(torch.load("./Models/FedProx/net.pt", weights_only=True))
    net.to(DEVICE)
    loss = nn.BCEWithLogitsLoss(torch.Tensor([1-x/sum(number_label) for x in(number_label)])).to(DEVICE)
    valid(net=net, valid_loader=testloader, e=0,lossf=loss, DEVICE=DEVICE)