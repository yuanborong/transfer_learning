import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def cos_dist(vec1,vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1=float(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    return dist1

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

csv_path = 'D:\\研究生\\transfer_learning\\result\\analysis_reason\\LR_coef\\'
transfer_csv_path = 'transfer_transitive\\'
without_transfer_csv_path = 'large_group_without_transfer\\'
cosine_similarity_csv_path = 'cosine_similarity_in_large_group\\'


for data_num in range(1, 6):

    transfer_csv_name = 'middle_coef_{}.csv'.format(data_num)
    without_transfer_csv_name = 'large_group_coef_{}.csv'.format(data_num)
    cosine_similarity_csv_name = 'cosine_similarity_in_large_group_model_{}.csv'.format(data_num)

    df_transfer = pd.read_csv(csv_path + transfer_csv_path + transfer_csv_name , index_col=0)
    df_without_transfer = pd.read_csv(csv_path + without_transfer_csv_path + without_transfer_csv_name , index_col=0)
    df_cosine_similarity = pd.DataFrame(np.ones((len(large_group_list), len(large_group_list))) * 0 , index=large_group_list ,
                                        columns = large_group_list)

    for i in range(len(large_group_list)):
        transfer_coef = np.array(df_transfer.loc[large_group_list[i] , :])
        without_transfer_coef = np.array(df_without_transfer.loc[large_group_list[i] , :])
        cos_value = cos_dist(transfer_coef , without_transfer_coef)

        df_cosine_similarity.loc[large_group_list[i] , large_group_list[i]] = cos_value
    # cosine_matrix = cosine_similarity(df_transfer.values , df_without_transfer.values)

    df_cosine_similarity.to_csv(csv_path + cosine_similarity_csv_path + cosine_similarity_csv_name)

    print('\nFinish data_' + str(data_num) + '.......\n\n')

print("Done........")