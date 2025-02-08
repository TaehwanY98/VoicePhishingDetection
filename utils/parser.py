import argparse


def FederatedParser():
    parser = argparse.ArgumentParser(prog="", description="")
    parser.add_argument("-v","--version", type=str, default="central")
    parser.add_argument("-s","--seed", type=int, default=2024)
    parser.add_argument("-r", "--round", type= int, default=30)
    parser.add_argument("--numClient", type= int, default=8)
    parser.add_argument("-e", '--epoch',type= int, default=3)
    parser.add_argument("--lr", type= float, default= 4e-3)
    args = parser.parse_args()
    return args