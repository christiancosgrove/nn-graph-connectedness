import os
import numpy as np
import argparse
from load import load_data


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='./data')
args = parser.parse_args()


load_data(args.data)