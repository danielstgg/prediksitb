import pickle
import numpy as np
import pandas as pd
#from sklearn.externals import joblib
#from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Data testing.csv')
df.index += 1

df.head()

# fungsi untuk menghilangkan data kosong
def clean_data(dataset):
    data = []
    for row in dataset:
        if row[1] != '0' and row[2] != '0' and row[3] != '0' and row[4] != '0' and row[5] != '0' and row[6] != '0' and row[7] != '0':       # cek data clean
            data.append(row)
    return data

# fungsi untuk menampilkan jumlah data, class 0, dan class 1 pada dataset
def check_dataset(dataset):
    total_data = 0
    class_0 = 0
    class_1 = 0

    for row in dataset:
        total_data = total_data + 1         # hitung jumlah data
        if row[8] == '1' :
            class_1 = class_1 + 1           # hitung jumlah data class 1
        else :
            class_0 = class_0 + 1           # hitung jumlah data class 0

# fungsi untuk normalisasi minmax
def min_max_norm(arr):
    # min-max norm
    data = minmax_scale(arr)
    for row in data:
        row[0] = round(row[0], 3)
        row[1] = round(row[1], 3)
        row[2] = round(row[2], 3)
        row[3] = round(row[3], 3)
        row[4] = round(row[4], 3)
        row[5] = round(row[5], 3)
        row[6] = round(row[6], 3)
        row[7] = round(row[7], 3)
        row[8] = round(row[8], 3)
        row[9] = round(row[9], 3)
    return data


    clean_dataset = clean_data(dataset)     # clean data
    #print("dataset clean")
    check_dataset(clean_dataset)
    #for row in clean_dataset :
    #     print(row)


    normalisasi = min_max_norm(clean_dataset)# normalisasi min max
    #print("dataset normalisasi")
    check_dataset(clean_dataset)

    normalisasi_pd = pd.DataFrame(normalisasi)# convert ke pandas
    # Inisialisasi fitur dan class
    X = normalisasi_pd.values[:, 0:8] 
    Y = normalisasi_pd.values[:, 8]

np_target = df['hasil'].values

np_data = df.iloc[:,:-1].values

#Naive Bayes
clf = GaussianNB()

#Split dataset, 20% testing 80% training.
X_train, X_test, y_train, y_test = train_test_split(np_data,np_target,test_size=0.3, random_state=None)

X_train

#Cuma buat liat ukuran data train vs data test
len(X_train), len(X_test)

#Fit model SVM sesuai data training (proses training model)
clf.fit(X_train, y_train)

#CrossValidation model yang sudah dibangun. Dengan parameter scoring=accuracy,precision, dan recall
scoring = ['accuracy', 'precision', 'recall']
scores = cross_validate(clf, X_test, y_test, scoring=scoring, cv=5, return_train_score=True)

#Menampilkan hasil dari cross validation
scores

#menampilkan test_accuracy dari cross validation, memiliki 5 value, tiap value adalah representasi akurasi dari tiap iterasi
scores['test_accuracy']

#Hasil cross validation berupa dictionary. Kemudian dictionary disimpan menjadi dataframe. Hanya untuk keperluan pembacaan
df_scores = pd.DataFrame.from_dict(scores)
df_scores.index +=1
df_scores

#DUMP MODEL
pickle.dump(clf, open('model_nb.pkl','wb'))

sample = np.array([0,0,0,1,0,1,0,1,0])
clf.predict_proba(sample.reshape(1,-1))

y_pred = clf.predict(np_data)
tn, fp, fn, tp = confusion_matrix(np_target, y_pred).ravel()

acc = accuracy_score(np_target, y_pred)
prec= precision_score(np_target, y_pred)
recall = recall_score(np_target, y_pred)

print(acc, prec, recall)