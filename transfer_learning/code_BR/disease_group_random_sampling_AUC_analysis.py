import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
# csv_path
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/'

for data_num in range(1 , 5):
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    # 生成不同的随机抽样比例
    sample_size = []
    for i in range(2 , 21):
        sample_size.append(i * 0.05)

    # 写入结果文件，题头（data_1 , data_2）
    # f_reuslt = open(txt_path, 'a+')
    # f_reuslt.write('data_{}_result'.format(data_num))
    # f_reuslt.write('\n')
    # f_reuslt.close()

    print('\nBegin data_' + str(data_num) +'.......\n\n')
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

        auc_list = []

        # 按不同的sample_size，df.sample进行随机抽样
        for frac in sample_size:
            # random sampling for test auc
            random_sampling_train_meaningful_sample = train_meaningful_sample.sample(frac=frac, axis=0)
            X_train = random_sampling_train_meaningful_sample.drop(['Label'], axis=1)
            y_train = random_sampling_train_meaningful_sample['Label']

            # build LR model for random sampling
            lr_DG_ran_smp = LogisticRegression(n_jobs=-1)
            lr_DG_ran_smp.fit(X_train, y_train)
            y_predict = lr_DG_ran_smp.predict(X_test)
            auc = roc_auc_score(y_test, y_predict)

            auc_dataframe.loc[disease_list.iloc[disease_num, 0] , frac] = round(auc , 6)
            # AUC保留6位小数
            auc_list.append(round(auc , 6))

        print('Sample size ' + str(frac) + ' has completed.........')

    csv_name = 'random_sampling_auc_result_data_{}.csv'.format(data_num)
    auc_dataframe.to_csv(csv_path + csv_name)

    print('\nFinish data_' + str(data_num) +'.......\n\n')


print("Done........")

# 每次写之前都进行读，那么久不会覆盖
        # auc结果写入文件
        # f_reuslt = open(txt_path, 'a+')
        # f_reuslt.write(disease_list.iloc[disease_num, 0] + ': ' )
        # f_reuslt.write(str(auc_list))
        # f_reuslt.write('\n')
        # f_reuslt.close()