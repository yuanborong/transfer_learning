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
# # number of trees based on source domain <= source_round
# source_round=60
# #number of trees based on target domain <= middle_round
# middle_round=20
#number of trees based on target domain <= target_round
# target_round=20


param_n_estimators_dict = {
    'Drg0' : [45 , 5 , 50] ,
    'Drg2' : [24 , 6 , 70] ,
    'Drg3' : [9 , 1 , 90] ,
    'Drg50': [4, 6 , 90],
    'Drg52': [56, 14 , 30],
    'Drg66': [5, 45 , 50],
    'Drg67': [18, 42 , 40],
    'Drg68': [4, 16 , 80],
    'Drg69': [6 , 54  , 40],
    'Drg84': [18, 42 , 40],
    'Drg96': [8, 2 , 90],
    'Drg97': [20, 20 , 60],
    'Drg178': [24, 56 , 20],
    'Drg179': [8, 72 , 20],
    'Drg256': [63, 27 , 10],
    'Drg259': [24, 36 , 40],
    'Drg261': [21, 49 , 30],
    'Drg262': [42, 28 , 30],
    'Drg263': [9, 81 , 10],
    'Drg309': [72, 18 , 10],
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
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/analysis_reason/small_sample_size/transfer_transitive(20%_large_group_10%_small_group)/GBM/'
# set data result csv's name
mean_auc_csv_name = 'transfer_transitive_from_all_data_mean.csv'
auc_by_source_model_csv_name = 'group_disease_data_by_source_model_with_all_data.csv'
auc_by_middle_model_csv_name = 'group_disease_data_by_middle_model_with_all_data.csv'

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
sample_size = [0.1 , 0.2 , 0.3 , 0.4 , 0.5]
columns_name = ['0.1-0.1' , '0.2-0.1' , '0.3-0.1' , '0.4-0.1' , '0.5-0.1']

# 创建一个5折交叉平均的df
auc_source_dataframe = pd.DataFrame(np.ones((len(disease_list), len(columns_name))) * 0, index=disease_list.iloc[:, 0],
                                  columns=columns_name)
auc_middle_dataframe = pd.DataFrame(np.ones((len(disease_list), len(columns_name))) * 0, index=disease_list.iloc[:, 0],
                                  columns=columns_name)
auc_target_dataframe = pd.DataFrame(np.ones((len(disease_list), len(columns_name))) * 0, index=disease_list.iloc[:, 0],
                                  columns=columns_name)

for data_num in range(1, 6):
    # set each data result csv's name
    csv_name = 'transfer_all_data_{}.csv'.format(data_num)
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    # 按不同的sample_size，df.sample进行随机抽样
    for disease_num in range(len(disease_list)):
        param_n_estimators_list = param_n_estimators_dict.get(disease_list.iloc[disease_num, 0])
        source_round = param_n_estimators_list[0]
        middle_round = param_n_estimators_list[1]
        target_round = param_n_estimators_list[2]

        # 根据当前的小亚组，寻找它对应的大亚组
        large_group_name = small_group_dict.get(disease_list.iloc[disease_num, 0])
        # 按照某一个大亚组，large_group_items表示这个大亚组对对应的所有小亚组（drg_range）
        large_group_items = large_group_dict.get(large_group_name)
        # 先找出当前小亚组对应的大亚组的所有样本
        large_group_disease_all_feature_true = get_true_sample(train_ori, large_group_items)
        large_group_disease_all_sample = train_ori.loc[large_group_disease_all_feature_true]
        large_group_disease_all_sample_index = large_group_disease_all_sample.index.tolist()

        # 在训练集中找到小亚组的所有样本
        target_train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        target_train_meaningful_sample = train_ori.loc[target_train_feature_true]
        small_group_disease_all_sample_index = target_train_meaningful_sample.index.tolist()

        # 在测试集中找到小亚组的所有样本
        target_test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        target_test_meaningful_sample = test_ori.loc[target_test_feature_true]
        X_test = target_test_meaningful_sample.drop(['Label'], axis=1)
        y_test = target_test_meaningful_sample['Label']

        # 先在源域中把所有的大亚组全部剔除
        source_expect_middle_sample_boolean_true = []
        for i in range(train_ori.shape[0]):
            if i in large_group_disease_all_sample_index:
                source_expect_middle_sample_boolean_true.append(False)
            else:
                source_expect_middle_sample_boolean_true.append(True)
        source_expect_middle_sample = train_ori.loc[source_expect_middle_sample_boolean_true]
        source_expect_middle_sample_index = source_expect_middle_sample.index.tolist()

        for frac_index in range(len(sample_size)):
            auc_by_source_model_list = []
            auc_by_middle_model_list = []
            auc_by_target_model_list = []
            i = 0
            while i < 10:
                # 对小亚组随机抽样
                small_group_disease_meaningful_sample = target_train_meaningful_sample.sample(frac=0.1, axis=0)
                target_domain_index = small_group_disease_meaningful_sample.index.tolist()

                # 首先处理中间域数据
                # 在大亚组里面首先剔除掉小亚组
                middle_expect_small_group_sample_boolean_true = (large_group_disease_all_sample.loc[:, disease_list.iloc[disease_num, 0]] == 0)
                middle_expect_small_group_sample = large_group_disease_all_sample.loc[middle_expect_small_group_sample_boolean_true]
                # 对剔除小亚组后的大亚组随机抽样
                middle_expect_small_group_sample_small_size = middle_expect_small_group_sample.sample(frac = sample_size[frac_index] , axis = 0)
                middle_expect_small_group_sample_small_size_index = middle_expect_small_group_sample_small_size.index.tolist()
                # 在随机抽样后的大亚组，加入上随机抽样的小亚组
                middle_domain_index = list(set(target_domain_index + middle_expect_small_group_sample_small_size_index))
                # 依据中间域下标Index，得到布尔列表
                middle_meaningful_sample_boolean_true = []
                for i in range(train_ori.shape[0]):
                    if i in middle_domain_index:
                        middle_meaningful_sample_boolean_true.append(True)
                    else:
                        middle_meaningful_sample_boolean_true.append(False)
                # 得到中间域数据样本
                middle_meaningful_sample = train_ori.loc[middle_meaningful_sample_boolean_true]
                middle_X_train = middle_meaningful_sample.drop(['Label'] , axis = 1)
                middle_y_train = middle_meaningful_sample['Label']

                # 再处理源域数据
                # 首先在源域中剔除掉大亚组数据（前面已经得到了：source_expect_middle_sample、source_expect_middle_sample_index）
                # 然后加上中间域数据就行了（中间域是由大亚组是20%和小亚组10%组成）
                source_domain_index = list(set(source_expect_middle_sample_index + middle_domain_index))
                # 依据源域下标Index，得到布尔列表
                source_meaningful_sample_boolean_true = []
                for i in range(train_ori.shape[0]):
                    if i in source_domain_index:
                        source_meaningful_sample_boolean_true.append(True)
                    else:
                        source_meaningful_sample_boolean_true.append(False)
                # 得到源域数据样本
                source_meaningful_sample = train_ori.loc[source_meaningful_sample_boolean_true]
                source_X_train = middle_meaningful_sample.drop(['Label'] , axis = 1)
                source_y_train = middle_meaningful_sample['Label']

                # 得到目标域数据
                target_X_train = small_group_disease_meaningful_sample.drop(['Label'], axis=1)
                target_y_train = small_group_disease_meaningful_sample['Label']

                # 得到最终的源域、中间域、目标域的所有符合的样本后，三次建模
                # 构建源域模型
                gbm_All = GradientBoostingClassifier(n_estimators=source_round, learning_rate=0.1, subsample=0.8,
                                                     loss='deviance',
                                                     max_features='sqrt', max_depth=3, min_samples_split=10,
                                                     min_samples_leaf=3,
                                                     min_weight_fraction_leaf=0, random_state=10)
                gbm_All.fit(source_X_train, source_y_train)
                gbm_source = gbm_init(gbm_All)
                # 利用源域模型预测小亚组
                y_predict_by_source_model = gbm_All.predict_proba(X_test)[:, 1]
                auc_by_source_model = roc_auc_score(y_test, y_predict_by_source_model)
                auc_by_source_model_list.append(auc_by_source_model)
                # 构建中间域模型
                gbm_large_group = GradientBoostingClassifier(init=gbm_source, n_estimators=middle_round,
                                                             learning_rate=0.1, subsample=0.8, loss='deviance',
                                                             max_features='sqrt', max_depth=3, min_samples_split=10,
                                                             min_samples_leaf=3, min_weight_fraction_leaf=0,
                                                             random_state=10)
                gbm_large_group.fit(middle_X_train, middle_y_train)
                gbm_middle = gbm_init(gbm_large_group)
                # 利用中间域模型预测小亚组
                y_predict_by_middle_model = gbm_large_group.predict_proba(X_test)[:, 1]
                auc_by_middle_model = roc_auc_score(y_test, y_predict_by_middle_model)
                auc_by_middle_model_list.append(auc_by_middle_model)

                # 构建目标域模型
                gbm_target = GradientBoostingClassifier(init=gbm_middle, n_estimators=target_round, learning_rate=0.1,
                                                        subsample=0.8, loss='deviance', max_features='sqrt',
                                                        max_depth=3, min_samples_split=10, min_samples_leaf=3,
                                                        min_weight_fraction_leaf=0, random_state=10)
                try:
                    gbm_target.fit(target_X_train, target_y_train)
                except Exception:
                    print('restart')
                    continue
                y_predict = gbm_target.predict_proba(X_test)[:, 1]
                auc_by_target_model = roc_auc_score(y_test, y_predict)
                auc_by_target_model_list.append(auc_by_target_model)
                i = i + 1

            auc_source_dataframe.loc[disease_list.iloc[disease_num, 0], columns_name[frac_index]] += np.mean(auc_by_source_model_list)
            auc_middle_dataframe.loc[disease_list.iloc[disease_num, 0], columns_name[frac_index]] += np.mean(auc_by_middle_model_list)
            auc_target_dataframe.loc[disease_list.iloc[disease_num, 0], columns_name[frac_index]] += np.mean(auc_by_target_model_list)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_source_dataframe = auc_source_dataframe.apply(lambda x: round(x / 5, 3))
auc_source_dataframe.to_csv(csv_path + auc_by_source_model_csv_name)
auc_middle_dataframe = auc_middle_dataframe.apply(lambda x: round(x / 5 , 3))
auc_middle_dataframe.to_csv(csv_path + auc_by_middle_model_csv_name)
auc_target_dataframe = auc_target_dataframe.apply(lambda x: round(x / 5, 3))
auc_target_dataframe.to_csv(csv_path + mean_auc_csv_name)

print("Done........")
