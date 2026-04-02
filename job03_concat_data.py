import pandas as pd
import glob
import datetime

data_path = glob.glob('./*.csv')
print(data_path)
df = pd.DataFrame()
for path in data_path:
    df_section = pd.read_csv(path, index_col=0)
    df = pd.concat([df, df_section])
df.info()
print(df.head())
print(df.category.value_counts())

df.to_csv('./news_titles_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), index=False)

















