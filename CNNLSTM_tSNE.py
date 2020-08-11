# -- coding: utf-8 --
"""
Created on Sun May 17 15:12:59 2020

@author: Gizem Çoban
"""

#Kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Verinin Yüklenmesi
veri=pd.read_csv("heart.csv")


#sınıf sayısını belirle
label_encoder=LabelEncoder().fit(veri.target)
labels=label_encoder.transform(veri.target)
classes=list(label_encoder.classes_)

#Eğitim verisinin oluşturulması
x=veri.drop(["target"], axis=1)
nb_features=13
nb_classes=len(classes)

#verilerin standartlaştırılması
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x.values)
veri=scaler.transform(x.values)

#Eğitim verisinin eğitim ve doğrulama için ayarlanması
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid=train_test_split(veri,labels,test_size=0.2)

#TSNE Analizi
import seaborn as sns
from sklearn.manifold import TSNE
Tsnemodel = TSNE(n_components=2, random_state=0)
tsne_obj= Tsnemodel .fit_transform(X_train)
tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                        'Y':tsne_obj[:,1],
                        'digit':y_train})
sns.scatterplot(x="X", y="Y",
              data=tsne_df);


#Etiketlerin Kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y_train=to_categorical(y_train)
y_valid=to_categorical(y_valid)

#Girdi verilerinin yeniden boyulandırılması
X_train=np.array(X_train).reshape(242,13,1)
X_valid=np.array(X_valid).reshape(61,13,1)


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
model.add(LSTM(512))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1024,activation="relu"))
model.add(Dense(1024,activation="relu"))
model.add(Dense(nb_classes,activation="softmax"))
model.summary()


#Modelin kesinlik,F1 Score vb hesaplanması
from keras import backend as K
from sklearn import metrics
import tensorflow as tf
def recall_m(y_true,y_pred):
    true_positivies=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    possible_positivies=K.sum(K.round(K.clip(y_true,0,1)))
    recall=true_positivies/(possible_positivies+K.epsilon())
    return recall
def precision_m(y_true,y_pred):
    true_positivies=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predictes_positivies=K.sum(K.round(K.clip(y_pred,0,1)))
    recall=true_positivies/(predictes_positivies+K.epsilon())
    return recall
def f1_m(y_true,y_pred):
    precision=precision_m(y_true,y_pred)
    recall=recall_m(y_true,y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def auc_m(y_true,y_pred):
    auc= tf.metrics.AUC(y_true,y_pred)
    return auc


#Modelin derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy",f1_m,precision_m,recall_m, tf.metrics.AUC(name="auc")])


#Modelin Eğitilmesi
score=model.fit(X_train,y_train,epochs=75,validation_data=(X_valid,y_valid))

#gerekli değerlerin gösterilmesi
print ("Ortalama Eğitim Kaybı:",np.mean(model.history.history["loss"]))
print ("Ortalama Eğitim Başarımı:",np.mean(model.history.history["accuracy"]))
print ("Ortalama Doğrulama Kaybı:",np.mean(model.history.history["val_loss"]))
print ("Ortalama F1-Score:",np.mean(model.history.history["val_f1_m"]))
print ("Ortalama Kesinlik:",np.mean(model.history.history["val_precision_m"]))
print ("Ortalama Duyarlılık:",np.mean(model.history.history["val_recall_m"]))
print ("Ortalama AucDeğer:",np.mean(model.history.history["auc"]))
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

#Duyarlılık Gösterimi
import matplotlib.pyplot as plt
plt.plot(model.history.history["recall_m"])
plt.plot(model.history.history["val_recall_m"])
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show

#Kesinlik Gösterimi
import matplotlib.pyplot as plt
plt.plot(model.history.history["precision_m"])
plt.plot(model.history.history["val_precision_m"])
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show


#F1-Score
import matplotlib.pyplot as plt
plt.plot(model.history.history["f1_m"],color="c")
plt.plot(model.history.history["val_f1_m"],color="k")
plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show



#Auc Gösterimi
import matplotlib.pyplot as plt
plt.plot(model.history.history["auc"],color="y")

plt.title("Model Kaybı")
plt.ylabel("Kayıp")
plt.xlabel("Epak")
plt.legend(["Eğitim","Test"],loc="upper left")
plt.show
# -- coding: utf-8 --