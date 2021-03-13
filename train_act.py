# from project.main.train_dstc_act import train as train_act
from project.main.train_jddc_act import train as train_act
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-fold', type=int)
parser.add_argument('--data', type=str, default='dstc8')
parser.add_argument('--model', type=str, default='HiGRU+ATTN')
args = parser.parse_args()

print('CUDA_VISIBLE_DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])
print('train data', args.data)
print('train model', args.model)
print('train fold', args.fold)

train_act(fold=args.fold, data_name=args.data, model_name=args.model)













