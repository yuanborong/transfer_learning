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
# number of trees based on source domain <= source_round
source_round=60
#number of trees based on target domain <= middle_round
middle_round=20
#number of trees based on target domain <= target_round
target_round=20

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
csv_path = '/home/huxinhou/WorkSpace_BR/transfer_learning/result/transfer_transitive/GBM/'
# set data result csv's name
mean_auc_csv_name = 'transfer_transitive_from_all_data_mean.csv'
auc_by_source_model_csv_name = 'group_disease_data_by_source_model_with_all_data.csv'
auc_by_middle_model_csv_name = 'group_disease_data_by_middle_model_with_all_data.csv'

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
sample_size = []
for i in range(2, 21):
    sample_size.append(i * 0.05)

# 创建一个5折交叉平均的df
auc_mean_dataframe = pd.DataFrame(np.ones((len(disease_list), len(sample_size))) * 0, index=disease_list.iloc[:, 0],
                                  columns=sample_size)
# 创建一个df记录 “ 2. 全局模型分别对各个亚组样本的AUC。”
auc_global_dataframe_columns = ['data_1' , 'data_2' , 'data_3' , 'data_4' , 'data_5' , 'mean_result']
auc_source_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=auc_global_dataframe_columns)
auc_middle_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=auc_global_dataframe_columns)

for data_num in range(1, 6):
    # set each data result csv's name
    csv_name = 'transfer_all_data_{}.csv'.format(data_num)
    # test data
    test_ori = pd.read_csv('/home/liukang/Doc/valid_df/test_{}.csv'.format(data_num))
    # training data
    train_ori = pd.read_csv('/home/liukang/Doc/valid_df/train_{}.csv'.format(data_num))

    X_train_all_data = train_ori.drop(['Label'], axis=1)
    y_train_all_data = train_ori['Label']

    gbm_All = GradientBoostingClassifier(n_estimators=source_round, learning_rate=0.1, subsample=0.8, loss='deviance',
                                         max_features='sqrt', max_depth=3, min_samples_split=10, min_samples_leaf=3,
                                         min_weight_fraction_leaf=0, random_state=10)
    gbm_All.fit(X_train_all_data, y_train_all_data)
    gbm_source = gbm_init(gbm_All)

    # 初始化一个新的auc_dataframe
    auc_dataframe = pd.DataFrame(index=disease_list.iloc[:, 0], columns=sample_size)

    for disease_num in range(len(disease_list)):
        # 根据当前的小亚组，寻找它对应的大亚组
        large_group_name = small_group_dict.get(disease_list.iloc[disease_num , 0])
        # 按照某一个大亚组，large_group_items表示这个大亚组对对应的所有小亚组（drg_range）
        large_group_items = large_group_dict.get(large_group_name)

        # 依据对应属于的大亚组，先进行全体数据迁移到大亚组，得到迁移知识(第一次迁移)
        # find patients with a certain disease in middle domain
        middle_train_feature_true = get_true_sample(train_ori, large_group_items)
        middle_train_meaningful_sample = train_ori.loc[middle_train_feature_true]
        middle_X_train = middle_train_meaningful_sample.drop(['Label'], axis=1)
        middle_y_train = middle_train_meaningful_sample['Label']
        gbm_large_group = GradientBoostingClassifier(init=gbm_source,n_estimators=middle_round, learning_rate=0.1,subsample=0.8,loss='deviance',max_features='sqrt',max_depth=3,min_samples_split=10,min_samples_leaf=3,min_weight_fraction_leaf=0,random_state=10)
        gbm_large_group.fit(middle_X_train , middle_y_train)
        gbm_middle = gbm_init(gbm_large_group)

        # find patients with a certain disease in target domain
        target_train_feature_true = train_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        target_train_meaningful_sample = train_ori.loc[target_train_feature_true]

        # get patients with small disease in test dataset (target domain's test sample)
        target_test_feature_true = test_ori.loc[:, disease_list.iloc[disease_num, 0]] > 0
        target_test_meaningful_sample = test_ori.loc[target_test_feature_true]
        X_test = target_test_meaningful_sample.drop(['Label'], axis=1)
        y_test = target_test_meaningful_sample['Label']

        # use source model to predict each group disease's AUC
        y_predict_by_source_model = gbm_All.predict_proba(X_test)[: , 1]
        auc_by_source_model = roc_auc_score(y_test , y_predict_by_source_model)
        auc_source_dataframe.loc[disease_list.iloc[disease_num , 0] , auc_global_dataframe_columns[data_num - 1]] = auc_by_source_model

        # use middle model to predict each group disease's AUC
        y_predict_by_middle_model = gbm_large_group.predict_proba(X_test)[:, 1]
        auc_by_middle_model = roc_auc_score(y_test, y_predict_by_middle_model)
        auc_middle_dataframe.loc[disease_list.iloc[disease_num, 0], auc_global_dataframe_columns[data_num - 1]] = auc_by_middle_model

        # 按不同的sample_size，df.sample进行随机抽样
        for frac in sample_size:
            auc_list = []
            i = 0
            while i < 10:
                # random sampling for test auc
                random_sampling_train_meaningful_sample = target_train_meaningful_sample.sample(frac=frac, axis=0)
                target_X_train = random_sampling_train_meaningful_sample.drop(['Label'], axis=1)
                target_y_train = random_sampling_train_meaningful_sample['Label']

                # # transfer to X_train
                # fit_train = X_train * Weight_importance_source_data
                # fit_train = fit_train * Weight_importance_from_middle_data

                # # build LR model for random sampling
                # lr_DG_ran_smp = LogisticRegression(n_jobs=-1)
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
                auc = roc_auc_score(y_test, y_predict)
                auc_list.append(auc)
                i = i + 1

            auc_dataframe.loc[disease_list.iloc[disease_num , 0], frac] = round(np.mean(auc_list), 3)
            auc_mean_dataframe.loc[disease_list.iloc[disease_num , 0], frac] += np.mean(auc_list)

    auc_dataframe.to_csv(csv_path + csv_name)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

auc_mean_dataframe = auc_mean_dataframe.apply(lambda x: round(x / 5, 3))
auc_mean_dataframe.to_csv(csv_path + mean_auc_csv_name)
auc_source_dataframe['mean_result'] = auc_source_dataframe[["data_1" , "data_2" , "data_3" , "data_4" , "data_5"]].mean(axis=1)
auc_source_dataframe.to_csv(csv_path + auc_by_source_model_csv_name)
auc_middle_dataframe['mean_result'] = auc_middle_dataframe[["data_1" , "data_2" , "data_3" , "data_4" , "data_5"]].mean(axis=1)
auc_middle_dataframe.to_csv(csv_path + auc_by_middle_model_csv_name)

print("Done........")
