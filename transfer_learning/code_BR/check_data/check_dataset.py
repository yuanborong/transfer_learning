import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# csv_path
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/analysis_reason/LR_coef/large_group_without_transfer/'

for data_num in range(1, 6):
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

print("Done........")
