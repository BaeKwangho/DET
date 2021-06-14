import sys
import argparse
import torch
import yaml

from data.dataset import get_dataloader

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Text Detection Training Session"
    )
    
    parser.add_argument(
        "-cf",
        "--config",
        type=str,
        help="Location of configuration yaml",
        default='./conf/train.yaml'
    )
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    with open(args.config) as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    
    dataset = get_dataloader(conf['Dataset'])
    
if __name__=='__main__':
    main()