import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

disease_list=pd.read_csv('/home/liukang/Doc/disease_top_31.csv')

for data_num in [1]:
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    sample_size = [np.linspace(0.05, 1, 20, endpoint=True)]

    # for disease_num in range(disease_list.shape[0]):
    for disease_num in [0]:
        # find patients with a certain disease
        train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        train_meaningful_sample = train_ori.loc[train_feature_true]
        # X_train = train_meaningful_sample.drop(['Label'], axis=1)
        # y_train = train_meaningful_sample['Label']

        test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        test_meaningful_sample = test_ori.loc[test_feature_true]
        X_test = test_meaningful_sample.drop(['Label'], axis=1)
        y_test = test_meaningful_sample['Label']

        sample_size = sample_size[:10]

        for frac in sample_size:
            auc_list = []

            # random sampling for test auc
            random_sampling_train_meaningful_sample = train_meaningful_sample.sample(frac=frac, axis=0)
            X_train = random_sampling_train_meaningful_sample.drop(['Label'], axis=1)
            y_train = random_sampling_train_meaningful_sample['Label']

            # build LR model for random sampling
            lr_DG_ran_smp = LogisticRegression(n_jobs=-1)
            lr_DG_ran_smp.fit(X_train, y_train)
            y_predict = lr_DG_ran_smp.predict(X_test)
            auc = roc_auc_score(y_test, y_predict)

            auc_list.append(auc)

        print(len(auc_list))
        print(auc_list)

