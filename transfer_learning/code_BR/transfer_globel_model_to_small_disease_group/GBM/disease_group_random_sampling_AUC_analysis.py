import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
# csv_path
# csv_path = '/home/liukang/Doc/transfer_learning/'
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/GBM/'

# 生成不同的随机抽样比例
sample_size = []
for i in range(2 , 21):
    sample_size.append(i * 0.05)

auc_mean_dataframe = pd.DataFrame( np.ones((len(disease_list),len(sample_size)))*0 , index=disease_list.iloc[: , 0] , columns=sample_size)

for data_num in range(1 , 6):
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    # 初始化一个新的auc_dataframe
    auc_dataframe = pd.DataFrame(index=disease_list.iloc[: , 0] , columns=sample_size)

    for disease_num in range(disease_list.shape[0]):
        # find patients with a certain disease
        train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        train_meaningful_sample = train_ori.loc[train_feature_true]
        # X_train = train_meaningful_sample.drop(['Label'], axis=1)
        # y_train = train_meaningful_sample['Label']

        test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        test_meaningful_sample = test_ori.loc[test_feature_true]
        X_test = test_meaningful_sample.drop(['Label'], axis=1)
        y_test = test_meaningful_sample['Label']

        # 按不同的sample_size，df.sample进行随机抽样
        for frac in sample_size:
            
            auc_list = []
            i = 0
            while i < 10:
                # random sampling for test auc
                random_sampling_train_meaningful_sample = train_meaningful_sample.sample(frac=frac, axis=0)
                X_train = random_sampling_train_meaningful_sample.drop(['Label'], axis=1)
                y_train = random_sampling_train_meaningful_sample['Label']
                
                # build GBM model for random sampling
                # lr_DG_ran_smp = LogisticRegression(n_jobs=-1)
                GBM_DG_ran_smp = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,random_state=10)
                try:
                    GBM_DG_ran_smp.fit(X_train, y_train)
                except Exception:
                    print('restart')
                    continue
                y_predict = GBM_DG_ran_smp.predict_proba(X_test)[: , 1]
                auc = roc_auc_score(y_test, y_predict)
                auc_list.append(auc)
                i=i+1

            auc_dataframe.loc[disease_list.iloc[disease_num, 0] , frac] = round(np.mean(auc_list) , 3)
            auc_mean_dataframe.loc[disease_list.iloc[disease_num, 0] , frac] += np.mean(auc_list)

    # auc_dataframe['mean_auc'] = np.mean(auc_dataframe.loc[])

    csv_name = 'random_sampling_auc_result_data_{}.csv'.format(data_num)
    auc_dataframe.to_csv(csv_path + csv_name)

    print('\nFinish data_' + str(data_num) +'.......\n\n')

auc_mean_dataframe = auc_mean_dataframe.apply(lambda x : round(x / 5 , 3))

mean_auc_csv_name = 'random_sampling_mean_auc_result_data.csv'
auc_mean_dataframe.to_csv(csv_path + mean_auc_csv_name)

print("Done........")
