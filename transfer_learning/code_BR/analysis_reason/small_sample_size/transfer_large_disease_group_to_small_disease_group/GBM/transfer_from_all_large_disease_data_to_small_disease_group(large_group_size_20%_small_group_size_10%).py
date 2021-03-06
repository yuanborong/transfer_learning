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

param_n_estimators_dict = {
    'Drg0' : [90 , 10],
    'Drg2' : [80 , 20],
    'Drg3' : [90 , 10],
    'Drg50': [40 , 60],
    'Drg52': [80 , 20],
    'Drg66': [10 , 90],
    'Drg67': [30 , 70],
    'Drg68': [20 , 80],
    'Drg69': [10 , 90],
    'Drg84': [30 , 70],
    'Drg96': [80 , 20],
    'Drg97': [50 , 50],
    'Drg178': [30 , 70],
    'Drg179': [10 , 90],
    'Drg256': [70 , 30],
    'Drg259': [40 , 60],
    'Drg261': [30 , 70],
    'Drg262': [60 , 40],
    'Drg263': [10 , 90],
    'Drg309': [80 , 20]
}
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
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/analysis_reason/small_sample_size/transfer_from_large_disease_group_to_small_disease_group(large_group_size_20%)/GBM/'
# set data result csv's name
auc_target_csv_name = 'transfer_from_all_data_to_small_disease_group_mean.csv'
auc_source_csv_name = 'group_disease_data_by_global_model_with_all_data.csv'

# get large disease group dict
large_group_dict = {
    "Liver and Gall" : ['Drg0' , 'Drg120' , 'Drg121' , 'Drg122' , 'Drg123' , 'Drg124' , 'Drg125' , 'Drg126' , 'Drg127', 'Drg128' , 'Drg129' , 'Drg130' ],
    "Blood" : ['Drg2' , 'Drg249' , 'Drg250' , 'Drg251' , 'Drg252' , 'Drg253' , 'Drg254' , 'Drg255' , 'Drg256' , 'Drg257' , 'Drg258' , 'Drg259' , 'Drg260'] ,
    "Lung" : ['Drg3' , 'Drg47' , 'Drg48' , 'Drg49' , 'Drg50' , 'Drg51' , 'Drg52' , 'Drg53' , 'Drg54' , 'Drg55' , 'Drg56' , 'Drg57' , 'Drg58' , 'Drg59' , 'Drg60' , 'Drg61' , 'Drg62' , 'Drg63'] ,
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
sample_size = [0.1]

# 创建一个5折交叉平均的df
auc_source_dataframe = pd.DataFrame(np.ones((len(disease_list), len(sample_size))) * 0, index=disease_list.iloc[:, 0],
                                  columns=sample_size)
auc_target_dataframe = pd.DataFrame(np.ones((len(disease_list), len(sample_size))) * 0, index=disease_list.iloc[:, 0],
                                  columns=sample_size)

for data_num in range(1, 6):
    # set each data result csv's name
    csv_name = 'transfer_all_data_{}.csv'.format(data_num)
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    for disease_num in range(len(disease_list)):
        param_n_estimators_list = param_n_estimators_dict.get(disease_list.iloc[disease_num, 0])
        source_round = param_n_estimators_list[0]
        target_round = param_n_estimators_list[1]

        # get patients with small disease in test dataset (target domain's test sample)
        test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        test_meaningful_sample = test_ori.loc[test_feature_true]
        X_test = test_meaningful_sample.drop(['Label'], axis=1)
        y_test = test_meaningful_sample['Label']

        # 根据当前的小亚组，寻找它对应的大亚组
        large_group_name = small_group_dict.get(disease_list.iloc[disease_num , 0])
        # 按照某一个大亚组，large_group_items表示这个大亚组对对应的所有小亚组（drg_range）
        large_group_items = large_group_dict.get(large_group_name)
        # 得到全部的大亚组（未剔除）
        source_train_feature_true = get_true_sample(train_ori , large_group_items)
        source_train_meaningful_sample = train_ori.loc[source_train_feature_true]

        # 得到全部的小亚组（训练集）
        target_train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        target_train_meaningful_sample = train_ori.loc[target_train_feature_true]

        # 先在大亚组样本中排除全部小亚组
        source_train_meaningful_true = (source_train_meaningful_sample.loc[:, disease_list.iloc[disease_num, 0]] == 0)
        source_train_meaningful_sample = source_train_meaningful_sample.loc[source_train_meaningful_true]

        # 按不同的sample_size，df.sample进行随机抽样
        for frac in sample_size:
            auc_source_list = []
            auc_target_list = []
            i = 0
            while i < 10:
                # 再在排除了小亚组的大亚组上，随机抽样
                source_train_small_simple_size = source_train_meaningful_sample.sample(frac = 0.2 , axis = 0)
                source_large_group_meaningful_index = source_train_small_simple_size.index.tolist()
                # 在小亚组上随机抽样
                target_train_small_simple_size = target_train_meaningful_sample.sample(frac=frac, axis=0)
                source_small_group_meaningful_index = target_train_small_simple_size.index.tolist()
                # 合并20%的大亚组和10%的小亚组(去重)
                source_train_meaningful_index = list(set(source_large_group_meaningful_index + source_small_group_meaningful_index))
                # 得到源域的布尔列表
                source_domain_index = []
                for i in range(train_ori.shape[0]):
                    if i in source_train_meaningful_index:
                        source_domain_index.append(True)
                    else:
                        source_domain_index.append(False)
                # 利用源域的布尔列表，得到源域样本
                source_train_sample = train_ori.loc[source_domain_index]
                source_X_train = source_train_sample.drop(['Label'] , axis = 1)
                source_y_train = source_train_sample['Label']
                # 构建源域模型
                # learn global model
                gbm_All = GradientBoostingClassifier(n_estimators=source_round, learning_rate=0.1, subsample=0.8,
                                                     loss='deviance',
                                                     max_features='sqrt', max_depth=3, min_samples_split=10,
                                                     min_samples_leaf=3,
                                                     min_weight_fraction_leaf=0, random_state=10)
                gbm_All.fit(source_X_train, source_y_train)
                # knowledge used for transfer
                gbm_ori = gbm_init(gbm_All)
                # 利用源域模型预测小亚组
                source_y_predice = gbm_All.predict_proba(X_test)[:, 1]
                auc_source = roc_auc_score(y_test , source_y_predice)
                auc_source_list.append(auc_source)

                # 得到当前小亚组小样本量的数据（真正的小样本量目标域）
                target_X_train = target_train_small_simple_size.drop(['Label'] , axis = 1)
                target_y_train = target_train_small_simple_size['Label']

                # 构建目标域模型
                # build GBM model for random sampling with tranfser
                gbm_DG_ran_smp = GradientBoostingClassifier(init=gbm_ori, n_estimators=target_round, learning_rate=0.1,
                                                            subsample=0.8, loss='deviance', max_features='sqrt',
                                                            max_depth=3, min_samples_split=10, min_samples_leaf=3,
                                                            min_weight_fraction_leaf=0, random_state=10)
                try:
                    gbm_DG_ran_smp.fit(target_X_train, target_y_train)
                except Exception:
                    print('restart')
                    continue
                target_y_predict = gbm_DG_ran_smp.predict_proba(X_test)[:, 1]
                auc_target = roc_auc_score(y_test, target_y_predict)
                auc_target_list.append(auc_target)
                i = i + 1

            auc_source_dataframe.loc[disease_list.iloc[disease_num, 0], frac] += np.mean(auc_source_list)
            auc_target_dataframe.loc[disease_list.iloc[disease_num, 0], frac] += np.mean(auc_target_list)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_source_dataframe = auc_source_dataframe.apply(lambda x: round(x / 5, 3))
auc_source_dataframe.to_csv(csv_path + auc_source_csv_name)
auc_target_dataframe = auc_target_dataframe.apply(lambda x: round(x / 5, 3))
auc_target_dataframe.to_csv(csv_path + auc_target_csv_name)

print("Done........")
