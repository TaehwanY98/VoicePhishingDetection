import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation
from flwr.client import NumPyClient
from creativeV2 import *
from collections import OrderedDict 
from torch import nn
from torch.optim.sgd import SGD
from flwr.common import Context
from utils import parser
import logging
eval_set = Voice_Fishing_Dataset("KorCCVi_v2.csv")
dataset = Voice_Fishing_Dataset("KorCCVi_v2.csv")
args = parser.FederatedParser()
eval_loader =DataLoader(eval_set, 8, shuffle=False, collate_fn=lambda x: x)
seeding(args)
class Client(NumPyClient):
    def __init__(self, net:nn.Module, epoch, train_loader, lossf, optimizer, DEVICE, trainF=train, validF=valid) -> None:
        super().__init__()
        self.net = net
        self.keys = net.state_dict().keys()
        self.epoch = epoch
        self.lossf = lossf
        self.optim = optimizer
        self.DEVICE=DEVICE
        self.train = trainF
        self.valid = validF
        self.train_loader = train_loader
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(self.net, train_loader=self.train_loader, valid_loader=None, epoch=self.epoch, lossf=self.lossf, optimizer=self.optim, DEVICE=self.DEVICE, validF=None, save_path=None)
        return self.get_parameters(config={}), len(self.train_loader), {}
    def set_parameters(self, parameters):
        params_dict = zip(self.keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

def client_fn(context: Context):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net=RNN(1,30,1,args.epoch,15000)
    net.to(DEVICE)
    trainset, _ = random_split(Voice_Fishing_Dataset("KorCCVi_v2.csv"), [0.2, 0.8])
    train_loader = DataLoader(trainset, 8, shuffle=True, collate_fn=lambda x: x)
    number_label = [dataset.normal, dataset.fishing]
    return Client(net, args.epoch, train_loader, nn.BCEWithLogitsLoss(torch.Tensor([1-x/sum(number_label) for x in(number_label)])).to(DEVICE), SGD(net.parameters(), lr=args.lr), DEVICE, train, valid).to_client()

def fl_save(server_round:int, parameters: fl.common.NDArrays, config, validF=valid):
    net = RNN(1,30, 1, args.epoch, 15000)
    set_parameters(net, parameters)
    save(net.state_dict(), f"./Models/FedAvg/net.pt")
    print("model is saved")
    return 0, {}

def fl_evaluate(server_round:int, parameters: fl.common.NDArrays, config, validF=valid):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RNN(1,30,1, args.epoch, 15000).to(DEVICE)
    set_parameters(net, parameters)
    number_label = [dataset.normal, dataset.fishing]
    hist=valid(net, eval_loader, 1, nn.BCEWithLogitsLoss(torch.Tensor([1-x/sum(number_label) for x in(number_label)])).to(DEVICE), DEVICE)
    save(net.state_dict(), f"./Models/FedAvg/net.pt")
    print("model is saved")
    return hist["loss"], {key:value for key, value in hist.items() if key !="loss"}

def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

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
    warnings.filterwarnings('ignore')
    
    hist = fl.simulation.start_simulation(
    client_fn=client_fn, # A function to run a _virtual_ client when required
    num_clients=args.numClient, # Total number of clients available
    config=fl.server.ServerConfig(num_rounds=args.round), # Specify number of FL rounds
    strategy=FedAvg(min_fit_clients=args.numClient, min_available_clients=args.numClient, min_evaluate_clients=args.numClient, evaluate_fn=fl_evaluate), # A Flower strategy
    client_resources = {"num_cpus": 3, "num_gpus": 1}
    )
    plt=pd.DataFrame(hist.losses_centralized)
    plt.to_csv(f"./CSV/FedAvg_loss.csv", index=False)
    pd.DataFrame(hist.metrics_centralized).to_csv(f"./CSV/FedAvg_metrics.csv", index=False)
    