import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 데이터 로드
df=pd.read_csv("Admission_Predict.csv")
X = df.iloc[:, 1:9]
Y = df.iloc[:, 9]

# 데이터 전처리
X_scaled= MinMaxScaler().fit_transform(X)

# 훈련셋, 검증셋, 시험셋
train_input, sub_input, train_target, sub_target = train_test_split(X_scaled, Y, test_size=0.3)
val_input, test_input, val_target, test_target = train_test_split(sub_input, sub_target, test_size=0.3)

# 모델 학습
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, activation='relu', input_dim=8))
    model.add(keras.layers.Dense(10,activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    return model

model = model_fn(keras.layers.Dropout(0.3))
model.compile(optimizer='nag', loss='binary_crossentropy', metrics='accuracy')
early_stopping_cb=keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
history=model.fit(train_input, train_target, epochs=100, batch_size=25, validation_data=(val_input, val_target), callbacks=[early_stopping_cb])

#model.summary()
print(early_stopping_cb.stopped_epoch)
# 모델 검증

model.evaluate(test_input, test_target)

# 모델 예측하기
XX=test_input
XX_scaled = MinMaxScaler().fit_transform(XX)

a=model.predict(XX_scaled, batch_size=25)

for i in range(0,len(test_input)):
    if a[i][0]>=0.5:
        print(a[i][0], "합격")
    else:
        print(a[i][0], "불합격")

import matplotlib.pyplot as plt

history_dict=history.history

acc=history_dict['accuracy']
val_acc=history_dict['val_accuracy']
loss=history_dict['loss']
val_loss=history_dict['val_loss']

epochs=range(1, len(loss)+1)

loss_ax = plt.subplot()
acc_ax = loss_ax.twinx()

loss_ax.plot(epochs, loss, 'y', label='Training loss')
loss_ax.plot(epochs, val_loss, 'r', label='validation loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(epochs, acc, 'b', label='Training acc')
acc_ax.plot(epochs, val_acc, 'g', label='validation acc')
acc_ax.legend(loc='upper right')
plt.show()
#################################################################
# 민감도 분석
# 검증셋 비율