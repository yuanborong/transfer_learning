import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# difine a pseudo-GBM to cheat sklearn's GBM
class gbm_init:
    def __init__(self, est):
        # post trained GBM
        self.est = est
    def predict(self, X):
        # return samples' score in trained GBM, don't use .predict() of .predict_proba() in sklearn 0.19.2
        return self.est._decision_function(X)
    def fit(self, X, y,sample_weight=None, **fit_params):
        # disable .fit() in GBM
        self.a=1+1

# number of trees based on source domain <= ori_round
ori_round=100

#number of trees based on target domain <= target_round
target_round=20

disease_list = pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
# csv_path
# csv_path = '/home/liukang/Doc/transfer_learning/'
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/GBM/'
#
mean_auc_csv_name = 'transfer_all_data_mean.csv'
auc_by_global_model_csv_name = 'group_disease_data_by_global_model_with_all_data.csv'

# 生成不同的随机抽样比例
sample_size = []
for i in range(2, 21):
    sample_size.append(i * 0.05)

# 创建一个5折交叉平均的df
auc_mean_dataframe = pd.DataFrame(np.ones((len(disease_list), len(sample_size))) * 0, index=disease_list.iloc[:, 0],
                                  columns=sample_size)
# 创建一个df记录 “ 2. 全局模型分别对各个亚组样本的AUC。”
auc_global_dataframe_columns = ['data_1' , 'data_2' , 'data_3' , 'data_4' , 'data_5' , 'mean_result']
auc_global_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=auc_global_dataframe_columns)

for data_num in range(1, 6):
    # set each data result csv's name
    csv_name = 'transfer_all_data_{}.csv'.format(data_num)
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))
    # print('\nBegin data_' + str(data_num) + '.......\n\n')

    X_train_all_data = train_ori.drop(['Label'], axis=1)
    y_train_all_data = train_ori['Label']

    # learn global model
    # lr_All = LogisticRegression(n_jobs=-1)
    gbm_All = GradientBoostingClassifier(n_estimators=ori_round, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,random_state=10)
    gbm_All.fit(X_train_all_data, y_train_all_data)

    # knowledge used for transfer
    gbm_ori = gbm_init(gbm_All)

    # 初始化一个新的auc_dataframe
    auc_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=sample_size)

    for disease_num in range(disease_list.shape[0]):
        # find patients with a certain disease
        train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        train_meaningful_sample = train_ori.loc[train_feature_true]

        test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        test_meaningful_sample = test_ori.loc[test_feature_true]
        X_test = test_meaningful_sample.drop(['Label'], axis=1)
        y_test = test_meaningful_sample['Label']
        # transfer to X_test
        # fit_test = X_test * Weight_importance_all_data

        # use global model to predict each group disease's AUC
        y_predict_by_global_model = gbm_All.predict_proba(X_test)[: , 1]
        auc_by_global_model = roc_auc_score(y_test , y_predict_by_global_model)
        auc_global_dataframe.loc[disease_list.iloc[disease_num , 0] , auc_global_dataframe_columns[data_num - 1]] = auc_by_global_model

        # 按不同的sample_size，df.sample进行随机抽样
        for frac in sample_size:
            auc_list = []
            i = 0
            while i < 10:
                # random sampling for test auc
                random_sampling_train_meaningful_sample = train_meaningful_sample.sample(frac=frac, axis=0)
                X_train = random_sampling_train_meaningful_sample.drop(['Label'], axis=1)
                y_train = random_sampling_train_meaningful_sample['Label']

                # transfer to X_train
                # fit_train = X_train * Weight_importance_all_data

                # build GBM model for random sampling with tranfser
                gbm_DG_ran_smp = GradientBoostingClassifier(init=gbm_ori,n_estimators=target_round, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,random_state=10)
                try:
                    gbm_DG_ran_smp.fit(X_train, y_train)
                except Exception:
                    print('restart')
                    continue
                y_predict = gbm_DG_ran_smp.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_predict)
                auc_list.append(auc)
                i = i + 1

            auc_dataframe.loc[disease_list.iloc[disease_num, 0], frac] = round(np.mean(auc_list), 3)
            auc_mean_dataframe.loc[disease_list.iloc[disease_num, 0], frac] += np.mean(auc_list)

    auc_dataframe.to_csv(csv_path + csv_name)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_mean_dataframe = auc_mean_dataframe.apply(lambda x: round(x / 5, 3))
auc_mean_dataframe.to_csv(csv_path + mean_auc_csv_name)
auc_global_dataframe['mean_result'] = auc_global_dataframe[["data_1" , "data_2" , "data_3" , "data_4" , "data_5"]].mean(axis=1)
auc_global_dataframe.to_csv(csv_path + auc_by_global_model_csv_name)

print("Done........")
