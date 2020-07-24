import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
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

# # number of trees based on source domain <= ori_round
# ori_round=100
#
# #number of trees based on target domain <= target_round
# target_round=20

# 传入数据集和要寻找的大亚组（多个疾病有一个满足即可）
def get_true_sample(dataframe , large_group_items):
    # np.zeros返回一个array
    train_feature_sum_in_large_group = np.zeros(dataframe.shape[0])
    for i in range(len(large_group_items)):
        train_feature_sum_in_large_group += np.array(dataframe.loc[: , large_group_items[i]].tolist())
    train_feature_sum_in_large_group = train_feature_sum_in_large_group.tolist()
    a = [(True if flag > 0 else False) for flag in train_feature_sum_in_large_group]
    return a

disease_list = pd.read_csv('/home/liukang/Doc/disease_top_20.csv')
# csv_path
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/analysis_reason/transfer_from_all_data_to_large_disease_group(use_large_group_model_predict_small_disease)/GBM/'
# set data result csv's name
mean_auc_csv_name_10 = 'transfer_from_all_data_to_large_disease_group_mean_10%.csv'
auc_by_global_model_csv_name_10 = 'group_disease_data_by_global_model_with_all_data_10%.csv'
mean_auc_csv_name_20 = 'transfer_from_all_data_to_large_disease_group_mean_20%.csv'
auc_by_global_model_csv_name_20 = 'group_disease_data_by_global_model_with_all_data_20%.csv'
mean_auc_csv_name_100 = 'transfer_from_all_data_to_large_disease_group_mean_100%.csv'
auc_by_global_model_csv_name_100 = 'group_disease_data_by_global_model_with_all_data_100%.csv'

# get large disease group dict
large_group_dict = {
    "Liver and Gall" : ['Drg120' , 'Drg121' , 'Drg122' , 'Drg123' , 'Drg124' , 'Drg125' , 'Drg126' , 'Drg127', 'Drg128' , 'Drg129' , 'Drg130' ],
    "Blood" : ['Drg249' , 'Drg250' , 'Drg251' , 'Drg252' , 'Drg253' , 'Drg254' , 'Drg255' , 'Drg256' , 'Drg257' , 'Drg258' , 'Drg259' , 'Drg260'] ,
    "Lung" : ['Drg47' , 'Drg48' , 'Drg49' , 'Drg50' , 'Drg51' , 'Drg52' , 'Drg53' , 'Drg54' , 'Drg55' , 'Drg56' , 'Drg57' , 'Drg58' , 'Drg59' , 'Drg60' , 'Drg61' , 'Drg62' , 'Drg63'] ,
    "Heart" : ['Drg64', 'Drg65', 'Drg66', 'Drg67', 'Drg68', 'Drg69', 'Drg70', 'Drg71', 'Drg72', 'Drg73', 'Drg74', 'Drg75', 'Drg76', 'Drg77', 'Drg78', 'Drg79', 'Drg80', 'Drg81', 'Drg82', 'Drg83', 'Drg84', 'Drg85', 'Drg86', 'Drg87', 'Drg88', 'Drg89', 'Drg90', 'Drg91', 'Drg92', 'Drg93', 'Drg94', 'Drg95'] ,
    "Digestive Tract" : ['Drg96', 'Drg97', 'Drg98', 'Drg99', 'Drg100', 'Drg101', 'Drg102', 'Drg103', 'Drg104', 'Drg105', 'Drg106', 'Drg107', 'Drg108', 'Drg109', 'Drg110', 'Drg111', 'Drg112', 'Drg113', 'Drg114', 'Drg115', 'Drg116', 'Drg117', 'Drg118', 'Drg119'] ,
    "Kidney" : ['Drg176', 'Drg177', 'Drg178', 'Drg179', 'Drg180', 'Drg181', 'Drg182', 'Drg183', 'Drg184', 'Drg185', 'Drg186', 'Drg187', 'Drg188', 'Drg189'] ,
    "Cancer" : ['Drg43','Drg55','Drg106','Drg127','Drg150','Drg162','Drg178','Drg195','Drg198','Drg205','Drg254', 'Drg255', 'Drg256', 'Drg257', 'Drg258', 'Drg259', 'Drg260'] ,
    "Systemic Infection" : ['Drg261', 'Drg262', 'Drg263', 'Drg264', 'Drg265', 'Drg266', 'Drg267'] ,
    "UNREL PDX" : ['Drg309', 'Drg310', 'Drg311']
}
large_group_list = list(large_group_dict.keys())

# connect small disease group to the large disease group
small_group_dict = {
    "Drg0" : 'Liver and Gall' ,
    "Drg2" : 'Blood',
    "Drg3" : 'Lung' ,
    "Drg50" : 'Lung' ,
    "Drg52" : 'Lung' ,
    "Drg66" : 'Heart' ,
    "Drg67" : 'Heart' ,
    "Drg68" : 'Heart' ,
    "Drg69" : 'Heart' ,
    "Drg84" : 'Heart' ,
    "Drg96" : 'Digestive Tract' ,
    "Drg97" : 'Digestive Tract' ,
    "Drg178" : 'Kidney' ,
    "Drg179" : 'Kidney' ,
    "Drg256" : 'Blood' ,
    "Drg259" : 'Cancer' ,
    "Drg261" : 'Systemic Infection' ,
    "Drg262" : 'Systemic Infection' ,
    "Drg263" : 'Systemic Infection' ,
    "Drg309" : 'UNREL PDX'
}

# ------------------------------------------------------------------------------------------------------

# 生成不同的随机抽样比例
sample_size = [0.1 , 0.2 , 1]
# for i in range(2, 21):
#     sample_size.append(i * 0.05)
source_n_estimators = []
for i in range(1 , 10):
    source_n_estimators.append(i * 10)

auc_mean_dataframe_10 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0, index=disease_list.iloc[:, 0],
                                  columns=source_n_estimators)
auc_global_dataframe_10 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0,
                                    index=disease_list.iloc[:, 0], columns=source_n_estimators)
auc_mean_dataframe_20 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0, index=disease_list.iloc[:, 0],
                                  columns=source_n_estimators)
auc_global_dataframe_20 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0,
                                    index=disease_list.iloc[:, 0], columns=source_n_estimators)
auc_mean_dataframe_100 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0, index=disease_list.iloc[:, 0],
                                  columns=source_n_estimators)
auc_global_dataframe_100 = pd.DataFrame(np.ones((len(disease_list), len(source_n_estimators))) * 0,
                                    index=disease_list.iloc[:, 0], columns=source_n_estimators)

for data_num in range(1, 6):
    # set each data result csv's name
    csv_name = 'transfer_all_data_{}.csv'.format(data_num)
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    # 初始化一个新的auc_dataframe
    auc_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=sample_size)

    for disease_num in range(len(disease_list)):
        # 根据当前的小亚组，寻找它对应的大亚组
        large_group_name = small_group_dict.get(disease_list.iloc[disease_num , 0])
        # 按照某一个大亚组，large_group_items表示这个大亚组对对应的所有小亚组（drg_range）
        large_group_items = large_group_dict.get(large_group_name)

        # 先找出当前小亚组对应的大亚组的所有样本
        large_group_disease_all_feature_true = get_true_sample(train_ori, large_group_items)
        large_group_disease_all_sample = train_ori.loc[large_group_disease_all_feature_true]
        large_group_disease_all_sample_index = large_group_disease_all_sample.index.tolist()

        # # 找出当前小亚组的所有样本（训练集）
        # small_group_disease_all_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        # small_group_disease_all_sample = train_ori.loc[small_group_disease_all_feature_true]
        # small_group_disease_all_sample_index = small_group_disease_all_sample.index.tolist()

        # get patients with small disease in test dataset (target domain's test sample)
        test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        test_meaningful_sample = test_ori.loc[test_feature_true]
        X_test = test_meaningful_sample.drop(['Label'], axis=1)
        y_test = test_meaningful_sample['Label']

        # 对大亚组进行随机抽样10%或20%
        for frac in sample_size:
            for source_estis in source_n_estimators:
                target_estis = 100 - source_estis

                auc_list = []
                i = 0
                while i < 10:
                    large_group_disease_meaningful_sample = large_group_disease_all_sample.sample(frac=frac, axis=0)
                    # 取随机抽样的大亚组样本下标，这些样本是保留的，其余的在大亚组样本是去掉
                    large_group_disease_meaningful_sample_index = large_group_disease_meaningful_sample.index.tolist()

                    # 得到不使用的大亚组样本下标
                    train_meaningful_false = []
                    for idx_all in large_group_disease_all_sample_index:
                        if idx_all not in large_group_disease_meaningful_sample_index:
                            train_meaningful_false.append(idx_all)

                    # 得到一个布尔向量，长度是训练集的样本量，标记保留哪些训练样本
                    train_meaningful_true = []
                    for i in range(train_ori.shape[0]):
                        if i not in train_meaningful_false:
                            train_meaningful_true.append(True)
                        else:
                            train_meaningful_true.append(False)

                    # 得到一个布尔向量，长度是训练集的样本量，标记保留哪些训练样本
                    # train_meaningful_true = []
                    # for i in range(train_ori.shape[0]):
                    #     if i in train_meaningful_true_index:
                    #         train_meaningful_true.append(True)
                    #     else:
                    #         train_meaningful_true.append(False)

                    train_meaningful_sample = train_ori.loc[train_meaningful_true]
                    source_X_train = train_meaningful_sample.drop(['Label'], axis=1)
                    source_y_train = train_meaningful_sample['Label']
                    gbm_All = GradientBoostingClassifier(n_estimators=source_estis, learning_rate=0.1, subsample=0.8,
                                                         loss='deviance',
                                                         max_features='sqrt', max_depth=3, min_samples_split=10,
                                                         min_samples_leaf=3,
                                                         min_weight_fraction_leaf=0, random_state=10)
                    gbm_All.fit(source_X_train, source_y_train)
                    # knowledge used for transfer
                    gbm_ori = gbm_init(gbm_All)

                    # use global model to predict each group disease's AUC
                    y_predict_by_global_model = gbm_All.predict_proba(X_test)[:, 1]
                    auc_by_global_model = roc_auc_score(y_test, y_predict_by_global_model)
                    if frac == 0.1:
                        auc_global_dataframe_10.loc[disease_list.iloc[disease_num , 0], source_estis] += auc_by_global_model
                    elif frac == 0.2:
                        auc_global_dataframe_20.loc[disease_list.iloc[disease_num , 0], source_estis] += auc_by_global_model
                    else:
                        auc_global_dataframe_100.loc[disease_list.iloc[disease_num , 0], source_estis] += auc_by_global_model

                    # 去目标域（大亚组）样本，这得到的是10%或20%的目标域样本
                    target_X_train = large_group_disease_meaningful_sample.drop(['Label'], axis=1)
                    target_y_train = large_group_disease_meaningful_sample['Label']
                    gbm_DG_ran_smp = GradientBoostingClassifier(init=gbm_ori, n_estimators=target_estis,
                                                                learning_rate=0.1,
                                                                subsample=0.8, loss='deviance', max_features='sqrt',
                                                                max_depth=3, min_samples_split=10, min_samples_leaf=3,
                                                                min_weight_fraction_leaf=0, random_state=10)
                    try:
                        gbm_DG_ran_smp.fit(target_X_train, target_y_train)
                    except Exception:
                        print('restart')
                        continue
                    y_predict = gbm_DG_ran_smp.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_predict)
                    auc_list.append(auc)
                    i = i + 1

                if frac == 0.1:
                    auc_mean_dataframe_10.loc[disease_list.iloc[disease_num , 0], source_estis] += np.mean(auc_list)
                elif frac == 0.2:
                    auc_mean_dataframe_20.loc[disease_list.iloc[disease_num , 0], source_estis] += np.mean(auc_list)
                else:
                    auc_mean_dataframe_100.loc[disease_list.iloc[disease_num , 0], source_estis] += np.mean(auc_list)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_mean_dataframe_10 = auc_mean_dataframe_10.apply(lambda x: round(x / 5, 3))
auc_mean_dataframe_10.to_csv(csv_path + mean_auc_csv_name_10)
auc_global_dataframe_10 = auc_global_dataframe_10.apply(lambda x: round(x / 5, 3))
auc_global_dataframe_10.to_csv(csv_path + auc_by_global_model_csv_name_10)
auc_mean_dataframe_20 = auc_mean_dataframe_20.apply(lambda x: round(x / 5, 3))
auc_mean_dataframe_20.to_csv(csv_path + mean_auc_csv_name_20)
auc_global_dataframe_20 = auc_global_dataframe_20.apply(lambda x: round(x / 5, 3))
auc_global_dataframe_20.to_csv(csv_path + auc_by_global_model_csv_name_20)
auc_mean_dataframe_100 = auc_mean_dataframe_100.apply(lambda x: round(x / 5, 3))
auc_mean_dataframe_100.to_csv(csv_path + mean_auc_csv_name_100)
auc_global_dataframe_100 = auc_global_dataframe_100.apply(lambda x: round(x / 5, 3))
auc_global_dataframe_100.to_csv(csv_path + auc_by_global_model_csv_name_100)

print("Done........")




