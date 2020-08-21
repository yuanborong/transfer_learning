import pandas as pd

csv_path = 'C://Users//xinho//Desktop//研二//transfer_learning//'
raw_distance_csv_name = 'distance.csv'

df_disease_list = pd.read_csv('C://Users//xinho//Desktop//研二//transfer_learning//disease_top_20.csv')
disease_list = df_disease_list.iloc[: , 0].tolist()

df_raw_distance = pd.read_csv('C://Users//xinho//Desktop//研二//transfer_learning//distance.csv' , index_col=0)
raw_column_name = df_raw_distance.columns.tolist()

for i in range(df_disease_list.shape[0]):
    csv_name = 'distance_' + str(disease_list[i]) + '.csv'
    df_new = df_raw_distance.iloc[i , :].to_frame()
    df_new = df_new.sort_index(by = [str(disease_list[i])] , axis = 0 , ascending = True)
    df_new.to_csv(csv_path + csv_name)

print("Done......")