import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import re

df = pd.read_csv('./news_titles_20260331.csv')
df.info()
print(df.head())

X = df.title
Y = df.category

encoder = LabelEncoder()
label_y = encoder.fit_transform(Y)
print(label_y[:5])
print(encoder.classes_)

with open('./encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(label_y)
print(onehot_y[:5])

okt = Okt()
okt_x = okt.morphs(X[0])
print(okt_x)
okt_x = okt.morphs(X[0], stem=True)
print(okt_x)

x = re.sub('[^가-힣]', ' ', X[2])
print(x)
# komoran = Komoran()
# komoran_x = komoran.morphs(X[0])
# print(komoran_x)
# komoran_x = komoran.pos(X[0])
# print(komoran_x)
print(len(X))
x = okt.morphs(X[0], stem=True)
print(x)
print(type(X))

X = list(X)
for i in range(len(X)):
    X[i] = re.sub('[^가-힣]', ' ', X[i])

    X[i] = okt.morphs(X[i], stem=True)

    if i % 100 == 0:
        print(i)
print(X[:10])
for idx, sentence in enumerate(X):
    words = []
    for word in sentence:
        if len(word) > 1:
            words.append(word)
    X[idx] = ' '.join(words)
print(X[:10])
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

tokened_x = tokenizer.texts_to_sequences(X)
print(tokened_x)
wordsize = len(tokenizer.word_counts) + 1
print(wordsize)

max = 0
for sentence in tokened_x:
    if max < len(sentence):
        max = len(sentence)
print(max)
with open('./token_max_{}.pkl'.format(max), 'wb') as f:
    pickle.dump(tokenizer, f)

x_pad = pad_sequences(tokened_x, maxlen=max)
print(x_pad[:10])

x_train, x_test, y_train, y_test = train_test_split(x_pad, onehot_y, test_size=0.1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
np.save('./dataset/title_x_train_wordsize{}.npy'.format(wordsize), x_train)
np.save('./dataset/title_x_test_wordsize{}.npy'.format(wordsize), x_test)
np.save('./dataset/title_y_train_wordsize{}.npy'.format(wordsize), y_train)
np.save('./dataset/title_y_test_wordsize{}.npy'.format(wordsize), y_test)









