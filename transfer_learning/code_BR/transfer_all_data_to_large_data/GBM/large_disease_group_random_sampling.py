import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# 传入数据集和要寻找的大亚组（多个疾病有一个满足即可）
def get_true_sample(dataframe , large_group_items):
    # np.zeros返回一个array
    train_feature_sum_in_large_group = np.zeros(dataframe.shape[0])
    for i in range(len(large_group_items)):
        train_feature_sum_in_large_group += np.array(dataframe.loc[: , large_group_items[i]].tolist())
    train_feature_sum_in_large_group = train_feature_sum_in_large_group.tolist()
    #train_feature_sum_in_large_group = train_feature_sum_in_large_group.apply(lambda x : 1 if x > 0 else 0)
    a = [(True if member > 0 else False) for member in train_feature_sum_in_large_group]
    return a

# get large disease group dict
large_group_dict = {
    "Liver and Gall" : ['Drg120' , 'Drg121' , 'Drg122' , 'Drg123' , 'Drg124' , 'Drg125' , 'Drg126' , 'Drg127', 'Drg128' , 'Drg129' , 'Drg130' ],
    "Blood" : ['Drg2' , 'Drg249' , 'Drg250' , 'Drg251' , 'Drg252' , 'Drg253' , 'Drg254' , 'Drg255' , 'Drg256' , 'Drg257' , 'Drg258' , 'Drg259' , 'Drg260'] ,
    "Lung" : ['Drg47' , 'Drg48' , 'Drg49' , 'Drg50' , 'Drg51' , 'Drg52' , 'Drg53' , 'Drg54' , 'Drg55' , 'Drg56' , 'Drg57' , 'Drg58' , 'Drg59' , 'Drg60' , 'Drg61' , 'Drg62' , 'Drg63'] ,
    "Heart" : ['Drg64', 'Drg65', 'Drg66', 'Drg67', 'Drg68', 'Drg69', 'Drg70', 'Drg71', 'Drg72', 'Drg73', 'Drg74', 'Drg75', 'Drg76', 'Drg77', 'Drg78', 'Drg79', 'Drg80', 'Drg81', 'Drg82', 'Drg83', 'Drg84', 'Drg85', 'Drg86', 'Drg87', 'Drg88', 'Drg89', 'Drg90', 'Drg91', 'Drg92', 'Drg93', 'Drg94', 'Drg95'] ,
    "Digestive Tract" : ['Drg96', 'Drg97', 'Drg98', 'Drg99', 'Drg100', 'Drg101', 'Drg102', 'Drg103', 'Drg104', 'Drg105', 'Drg106', 'Drg107', 'Drg108', 'Drg109', 'Drg110', 'Drg111', 'Drg112', 'Drg113', 'Drg114', 'Drg115', 'Drg116', 'Drg117', 'Drg118', 'Drg119'] ,
    "Kidney" : ['Drg176', 'Drg177', 'Drg178', 'Drg179', 'Drg180', 'Drg181', 'Drg182', 'Drg183', 'Drg184', 'Drg185', 'Drg186', 'Drg187', 'Drg188', 'Drg189'] ,
    "Cancer" : ['Drg43','Drg55','Drg106','Drg127','Drg150','Drg162','Drg178','Drg195','Drg198','Drg205','Drg254', 'Drg255', 'Drg256', 'Drg257', 'Drg258', 'Drg259', 'Drg260'] ,
    "Systemic Infection" : ['Drg261', 'Drg262', 'Drg263', 'Drg264', 'Drg265', 'Drg266', 'Drg267'] ,
    "UNREL PDX" : ['Drg309', 'Drg310', 'Drg311']
}
large_group_list = list(large_group_dict.keys())

# csv_path
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/transfer_from_all_data_to_large_disease_group/GBM/'

# 生成不同的随机抽样比例
sample_size = []
for i in range(2, 21):
    sample_size.append(i * 0.05)

# 创建一个5折交叉平均的df
auc_mean_dataframe = pd.DataFrame(np.ones((len(large_group_list), len(sample_size))) * 0, index=large_group_list,
                                  columns=sample_size)

for data_num in range(1, 6):
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    # 初始化一个新的auc_dataframe
    auc_dataframe = pd.DataFrame(index=large_group_list, columns=sample_size)

    for disease_num in range(len(large_group_list)):
        # 按照某一个大亚组，large_group_items表示这个大亚组对对应的所有小亚组（drg_range）
        large_group_items = large_group_dict.get(large_group_list[disease_num])

        # find patients with a certain disease
        train_feature_true = get_true_sample(train_ori, large_group_items)
        train_meaningful_sample = train_ori.loc[train_feature_true]

        test_feature_true = get_true_sample(test_ori, large_group_items)
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
                GBM_DG_ran_smp = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,random_state=10)
                try:
                    GBM_DG_ran_smp.fit(X_train, y_train)
                except Exception:
                    print('restart')
                    continue
                y_predict = GBM_DG_ran_smp.predict_proba(X_test)[: , 1]
                auc = roc_auc_score(y_test, y_predict)
                auc_list.append(auc)
                i = i + 1

            auc_dataframe.loc[large_group_list[disease_num], frac] = round(np.mean(auc_list), 3)
            auc_mean_dataframe.loc[large_group_list[disease_num], frac] += np.mean(auc_list)

    csv_name = 'random_sampling_auc_result_data_{}.csv'.format(data_num)
    auc_dataframe.to_csv(csv_path + csv_name)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_mean_dataframe = auc_mean_dataframe.apply(lambda x: round(x / 5, 3))

mean_auc_csv_name = 'random_sampling_mean_auc_result_data.csv'
auc_mean_dataframe.to_csv(csv_path + mean_auc_csv_name)

print("Done........")
