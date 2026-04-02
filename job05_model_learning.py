import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

x_train = np.load('./dataset/title_x_train_wordsize13613.npy', allow_pickle=True)
y_train = np.load('./dataset/title_y_train_wordsize13613.npy', allow_pickle=True)
x_test = np.load('./dataset/title_x_test_wordsize13613.npy', allow_pickle=True)
y_test = np.load('./dataset/title_y_test_wordsize13613.npy', allow_pickle=True)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(Embedding(13613, 300))
model.build(input_shape=(None, 32))
model.add(Conv1D(64, 5, padding='same', activation='relu'))
model.add(MaxPooling1D(1))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
score = model.evaluate(x_test, y_test, verbose=0)
print(score[1])
model.save('news_section_classfication_model_{}.keras'.format(score[1]))
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend(loc='lower right')
plt.show()














