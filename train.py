# from project.shared_private.train_dstc import train as train_dstc
# from project.shared_private.train_jddc import train as train_jddc
# from project.shared_private.train_ccpe import train as train_ccpe
# from project.shared_private.train_hierarchical_dstc import train as train_hierarchical_dstc
# from project.shared_private.train_hierarchical_jdcc import train as train_hierarchical_jdcc
# from project.shared_private.train_hierarchical_ccpe import train as train_hierarchical_ccpe

# train_dstc()
# train_jddc()
# train_ccpe()

# train_hierarchical_dstc()
# train_hierarchical_jdcc()
# train_hierarchical_ccpe()
# from project.main.train_dstc import train
from project.main.train_jddc import train
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

train(fold=args.fold, data_name=args.data, model_name=args.model)



