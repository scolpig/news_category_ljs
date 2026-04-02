from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime

category= ['Politics','Economics','Social','Culture','World','IT']
df_titles = pd.DataFrame()
for i in range(6):
    url  = "https://news.naver.com/section/10{}".format(i)

    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    title_tags = soup.select('.sa_text_strong')
    titles =[]
    for title_tag in title_tags:
        titles.append(title_tag.text)
    print(titles)
    df_section_titles = pd.DataFrame(titles,columns=['title'])
    df_section_titles['category'] = category[i]
    df_titles = pd.concat([df_titles,df_section_titles], ignore_index=True)
print(df_titles.head())
print(df_titles.info())
print(df_titles['category'].value_counts())
df_titles.to_csv('./crawling_data/naver_headline_news_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')))