# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:12:59 2020

@author: Gizem Çoban
"""

#Kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Verinin Yüklenmesi
veri=pd.read_csv("telefon_fiyat_değişimi.csv")

#sınıf sayısını belirle
label_encoder=LabelEncoder().fit(veri.price_range)
labels=label_encoder.transform(veri.price_range)
classes=list(label_encoder.classes_)


#Eğitim verisinin oluşturulması
x=veri.drop(["price_range"], axis=1)
nb_features=20
nb_classes=len(classes)

#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x.values)
veri=scaler.transform(x.values)

#Eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(veri,labels,test_size=0.2)


#Etiketlerin Kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)

#Girdi verilerinin yeniden boyulandırılması
X_train=np.array(X_train).reshape(1600,20,1)
X_valid=np.array(X_valid).reshape(400,20,1)

#Modelin oluşturulması
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout,Flatten,LSTM,Conv1D,MaxPooling1D,SimpleRNN

model=Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(256,1))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(SimpleRNN(512))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()

#Modelin derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])


#Modelin Eğitilmesi
score=model.fit(X_train,y_train,epochs=75,validation_data=(X_valid,y_valid))

#gerekli değerlerin gösterilmesi
print ("Ortalama Eğitim Kaybı:",np.mean(model.history.history["loss"]))
print ("Ortalama Eğitim Başarımı:",np.mean(model.history.history["accuracy"]))
print ("Ortalama Doğrulama Kaybı:",np.mean(model.history.history["val_loss"]))

#Eğitim ve Doğrulama Başarımlarının Gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Başarımı")
plt.ylabel("Başarım")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show


#Eğitim ve Doğrulama Kayıplarının Gösterilmesi
import matplotlib.pyplot as plt
plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show
