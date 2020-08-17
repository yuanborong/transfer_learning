import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def calculate_mean_distance(df_x , df_y):

    # df_x_index_list = df_x.index.tolist()
    # df_y_index_list = df_y.index.tolist()
    distance1 = 0.0

    for i in range(df_x.shape[0]):
        # print(df_y.shape[0])
        sample_x = df_x.iloc[i , :]
        distance2 = 0.0
        for j in range(df_y.shape[0]):
            sample_y = df_y.iloc[j , :]
            distance2 += np.linalg.norm(sample_x - sample_y)
        distance1 += (distance2 / df_y.shape[0])

    return (distance1 / df_x.shape[0])


disease_list = pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
# csv_path
# csv_path = '/home/liukang/Doc/transfer_learning/'
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/analysis_disease_group_distance/'
#
# mean_auc_csv_name = 'transfer_all_data_mean.csv'
# auc_by_global_model_csv_name = 'group_disease_data_by_global_model_with_all_data.csv'

# 创建一个df记录 “ 2. 全局模型分别对各个亚组样本的AUC。”
dataframe_columns_name_list = []
for i in range(315):
    columns_name = 'Drg' + str(i)
    dataframe_columns_name_list.append(columns_name)

distance_dataframe = pd.DataFrame(index=disease_list.iloc[: , 0], columns=dataframe_columns_name_list)

for data_num in range(1, 2):
    # set each data result csv's name
    csv_name = 'distance.csv'
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    # print('\nBegin data_' + str(data_num) + '.......\n\n')

    X_train_all_data = train_ori.drop(['Label'], axis=1)
    y_train_all_data = train_ori['Label']

    for disease_num_x in range(disease_list.iloc[: , 0]):
        # find patients with a certain disease
        train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num_x, 0]] > 0
        sample_x = train_ori.loc[train_feature_true]

        for disease_num_y in range(315):
            if disease_num_y == 0:
                continue
            drg_name = 'Drg' + str(disease_num_y)
            train_feature_true_y = train_ori.loc[: , drg_name] > 0
            sample_y = train_ori.loc[train_feature_true_y]
            if sample_y.shape[0] == 0 :
                continue

            sample_x_y_mean_distance = calculate_mean_distance(sample_x , sample_y)
            distance_dataframe.loc[disease_list.iloc[disease_num_x, 0] , drg_name] = sample_x_y_mean_distance

        print("Finish " + str(disease_list.iloc[disease_num_x, 0]) + ".....")

    distance_dataframe.to_csv(csv_path + csv_name)


print("Done........")
