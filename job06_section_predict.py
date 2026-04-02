import pickle
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

df = pd.read_csv('./crawling_data/naver_headline_news_20260402.csv')
df.columns = ['titles', 'category']
print(df.head())
df.info()
df.loc[df.category == 'Eoconomics', 'category'] = 'Economics'
print(df.category.value_counts())
X = df.titles
Y = df.category

with open('./encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
one_hot_y = to_categorical(labeled_y)

okt = Okt()
X = list(X)
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])
    X[i] = okt.morphs(X[i], stem=True)

for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:
            words.append(word)
    X[idx] = ' '.join(words)

with open('./token_max_32.pkl', 'rb') as f:
    token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 32:
        tokened_x[i] = tokened_x[i][:32]
x_pad = pad_sequences(tokened_x, maxlen=32)

model = load_model('./news_section_classfication_model_0.698358416557312.keras')
preds = model.predict(x_pad)
score = model.evaluate(x_pad, one_hot_y, verbose=0)
print(score[1])
print(preds[:6])

predixt_section = []
for pred in preds:
    most = label[np.argmax(pred)]
    predixt_section.append(most)
df['predict'] = predixt_section
print(df.head(30))

df['OX'] = 0
df.loc[df.category == df.predict, 'OX'] = 1
print(df.OX.mean())





























