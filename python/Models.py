#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import seaborn as sn
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


# In[99]:


np.random.seed(1)


# ### Датасет

# In[100]:


df = pd.read_csv("../data/train_features.csv", sep="|")
test = pd.read_csv("../data/test_features.csv", sep="|")
df.head()


# In[101]:


pd.set_option('display.max_rows', 200)
threshold = 1
# удаляем столбцы с маленьким стандартным отклонением
cols = df[df['is_insider'] == 1].std()[df[df['is_insider'] == 1].std() < threshold].index.values.tolist()
# не трогаем таргет столбец
cols.remove("is_insider")
df.drop(cols, axis=1, inplace=True)
# удаляем столбцы с маленьким стандартным отклонением
cols = df[df['is_insider'] == 0].std()[df[df['is_insider'] == 0].std() < threshold].index.values.tolist()
# не трогаем таргет столбец
cols.remove("is_insider")
df.drop(cols, axis=1, inplace=True)


# In[102]:


df.head()


# In[103]:


y = df['is_insider']
x_train, x_test, y_train, y_test = train_test_split(df, y, random_state = 0, test_size = 0.3)
X_test = test
Y_test = test['is_insider']


# In[104]:


medians = {} # словарь для хранения медиан для каждого столбца
# заполнение пустот в обучаемой выборке с помощью медиан
# и их использование для заполнения пустот в валидационной выборке
for col in x_train.columns:
    if col != 'is_insider':
        # медианы         
        medians[col + "_non_insider"] = x_train.loc[(x_train[col] != 0) & (x_train['is_insider'] == 0), col].median()
        medians[col + "_is_insider"] = x_train.loc[(x_train[col] != 0) & (x_train['is_insider'] == 1), col].median()
        # заполнение пустот в обучаемой выборке с помощью медиан из обучаемой выборки
        x_train.loc[(x_train[col] == 0) & (x_train['is_insider'] == 0), col] = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0 
        x_train.loc[(x_train[col] == 0) & (x_train['is_insider'] == 1), col] = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0
        # заполнение пустот в валидационной выборке с помощью медиан из обучаемой выборки
        x_test.loc[(x_test[col] == 0) & (x_test['is_insider'] == 0), col] = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0
        x_test.loc[(x_test[col] == 0) & (x_test['is_insider'] == 1), col] = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0
        # заполнение пустот в тестовой выборке с помощью медиан из обучаемой выборки
        X_test.loc[(X_test[col] == 0) & (X_test['is_insider'] == 0), col] = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0
        X_test.loc[(X_test[col] == 0) & (X_test['is_insider'] == 1), col] = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0
        
# очистка памяти
del medians


# In[105]:


medians_insider = []
medians_non_insider = []
cols = []
special_keys = [
    "Key.esc",
    "Key.tab",
    "Key.caps_lock",
    "Key.shift",
    "Key.ctrl",
    "Key.alt",
    "Key.cmd",
    "Key.space",
    "Key.enter",
    "Key.backspace",
]
for col in x_train.columns:
    # не считаем медианы для таргета и частотных признаков
    if col != 'is_insider' and col not in special_keys:
        cols.append(col)
        medians_insider.append(x_train[x_train['is_insider'] == 1][col].median())
        medians_non_insider.append(x_train[x_train['is_insider'] == 0][col].median())   


# In[106]:


rels = {
    "dwell": [0,0], # длительность нажатия первой буквы
    "interval": [0,0], # промежуток между отпусканием первой и нажатием второй буквы
    "flight": [0,0], # промежуток между нажатием первой и нажатием второй буквы
    "up_to_up": [0,0], # промежуток между отпусканием первой и отпусканием второй
    "dwell_first": [0,0], # длительность нажатия первой буквы
    "dwell_second": [0,0], # длительность нажатия второй буквы
    "dwell_third": [0,0], # длительность нажатия третьей буквы
    "interval_first": [0,0], # промежуток между отпусканием первой и нажатием второй буквы
    "interval_second": [0,0], # промежуток между отпусканием второй и нажатием третьей буквы
    "flight_first": [0,0], # промежуток между нажатием первой и нажатием второй буквы
    "flight_second": [0,0], # промежуток между нажатием второй и нажатием третьей буквы    
    "up_to_up_first": [0,0], # промежуток между отпусканием первой и отпусканием второй
    "up_to_up_second": [0,0], # промежуток между отпусканием второй и отпусканием третьей
    "latency": [0,0], # промежуток между нажатием первой и отпусканием третьей
}
for i in range(len(medians_insider)):
    for k,v in rels.items():
        if k in cols[i]:
            if medians_insider[i] < medians_non_insider[i]:
                v[0] += 1
            v[1] += 1

for k,v in rels.items():
    print(k,v[0],v[1],(v[1] - v[0]) / v[1] * 100 if v[1]-v[0] != 0 else 0)


# In[107]:


x_train.head()


# In[108]:


x_train_anomaly = pd.concat([x_train[x_train['is_insider'] == 0], x_test[x_test['is_insider'] == 0]]) 
y_train_anomaly = pd.concat([y_train[y_train == 0], y_test[y_test == 0]])
x_test_anomaly = pd.concat([x_train[x_train['is_insider'] == 1], x_test[x_test['is_insider'] == 1], 
    X_test[X_test['is_insider'] == 1], X_test[X_test['is_insider'] == 0]])
y_test_anomaly = pd.concat([y_train[y_train == 1],
    y_test[y_test == 1], Y_test[Y_test == 1], Y_test[Y_test == 0]])
len(x_train_anomaly.index), len(y_train_anomaly.index), len(x_test_anomaly.index), len(y_test_anomaly.index)


# In[109]:


# удаление столбца is_insider из x_train и x_test
x_train.drop(columns = ['is_insider'], inplace = True)
x_test.drop(columns = ['is_insider'], inplace = True)
X_test.drop(columns = ['is_insider'], inplace = True)
x_train_anomaly.drop(columns = ['is_insider'], inplace = True)
x_test_anomaly.drop(columns = ['is_insider'], inplace = True)


# ### Графики

# In[110]:


def plot_medians(name, medians_insider, medians_non_insider, labels):
    plt.figure(figsize=(30,20))
    plt.plot(medians_insider, '--rx', label = 'anomalous behaviour')
    plt.plot(medians_non_insider, '--bo', label = 'normal behaviour')
    plt.legend()
    plt.xticks(np.arange(len(labels)), labels=labels, rotation=90)
    plt.ylabel('Duration, ms')
    plt.xlabel('Features')
    plt.savefig(name, bbox_inches='tight')
    plt.show()


# In[111]:


def plot3d_samples(train, test, cols):
    x_train_insider = train[train['is_insider'] == 1][cols[0]].values
    y_train_insider = train[train['is_insider'] == 1][cols[1]].values
    z_train_insider = train[train['is_insider'] == 1][cols[2]].values
    
    x_train_non_insider = train[train['is_insider'] == 0][cols[0]].values
    y_train_non_insider = train[train['is_insider'] == 0][cols[1]].values
    z_train_non_insider = train[train['is_insider'] == 0][cols[2]].values
    
    x_test_insider = test[test['is_insider'] == 1][cols[0]].values
    y_test_insider = test[test['is_insider'] == 1][cols[1]].values
    z_test_insider = test[test['is_insider'] == 1][cols[2]].values
    
    x_test_non_insider = test[test['is_insider'] == 0][cols[0]].values
    y_test_non_insider = test[test['is_insider'] == 0][cols[1]].values
    z_test_non_insider = test[test['is_insider'] == 0][cols[2]].values
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ti = ax.scatter(x_train_insider, y_train_insider, z_train_insider, c='r', marker='o', label="Аномальное поведение в обучаемой выборке")
    tni = ax.scatter(x_train_non_insider, y_train_non_insider, z_train_non_insider, c='g', marker='o', label="Нормальное поведение в обучаемой выборке")
    Ti = ax.scatter(x_test_insider, y_test_insider, z_test_insider, c='r', marker='^', label="Аномальное поведение в тестовой выборке")
    Tni = ax.scatter(x_test_non_insider, y_test_non_insider, z_test_non_insider, c='g', marker='^', label="Нормальное поведение в тестовой выборке")

    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    # ax.legend()
    
    plt.title("Распределение примеров")
    
    plt.show()


# In[112]:


def plot3d(name, features, truth, preds, cols, anomaly=True):
    x = features[cols[0]]
    y = features[cols[1]]
    z = features[cols[2]]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    if anomaly:
        is_insider = -1
        non_insider = 1
    else:
        is_insider = 1
        non_insider = 0
    
    # calculate cases
    tn = np.logical_and(preds==is_insider, truth==is_insider)
    tp = np.logical_and(preds==non_insider, truth==non_insider)
    fn = np.logical_and(preds==is_insider, truth==non_insider)
    fp = np.logical_and(preds==non_insider, truth==is_insider)
    
    ax.scatter(x[tp], y[tp], z[tp],
               marker="o", edgecolors="lime", c="blue", lw=3, s=80, label="ИП")
    ax.scatter(x[tn], y[tn], z[tn],
               marker="o", edgecolors="red", c="peru", lw=3, s=80, label="ИО",)
    ax.scatter(x[fn], y[fn], z[fn],
               marker="o", edgecolors="red", c="blue", lw=3, s=80, label="ЛО")
    ax.scatter(x[fp], y[fp], z[fp],
               marker="o", edgecolors="lime", c="peru", lw=3, s=80, label="ЛП")
    
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])
    ax.set_zlabel(cols[2])
    ax.legend()
    
    plt.title(name)
    plt.show()


# ### Выделение признаков

# In[113]:


# график изначального относительного распределения 
# медиан признаков нарушителя и не нарушителя

# первая половина признаков
plot_medians(
    "features_1.png",
    medians_insider[:len(medians_insider) // 2], 
    medians_non_insider[:len(medians_non_insider) // 2],
    cols[:len(cols) // 2]
)
# вторая половина признаков
plot_medians(
    "features_2.png",
    medians_insider[len(medians_insider) // 2:], 
    medians_non_insider[len(medians_non_insider) // 2:],
    cols[len(cols) // 2:]
)


# In[81]:


transformer = SelectKBest(chi2, k = 3)
# transformer = SelectKBest(f_classif, k = 3)
X_new = transformer.fit_transform(x_train, y_train)

selected_columns = x_train.columns[transformer.get_support()].values
print("Выделенные признаки:", selected_columns)

x_train = x_train[selected_columns]
x_test = x_test[selected_columns]
X_test = X_test[selected_columns]

x_train_anomaly = x_train_anomaly[selected_columns]
x_test_anomaly = x_test_anomaly[selected_columns]

plot3d_samples(pd.concat([pd.concat([x_train, x_test]), pd.concat([y_train, y_test])], axis=1), 
       pd.concat([X_test, Y_test], axis=1), selected_columns)


# ### Метрики 

# In[ ]:


def report(clf, x_train, y_train, x_test, y_test, X_test, Y_test):
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    y_pred_train = clf.predict(x_train)
    Y_pred = clf.predict(X_test)
    # plot the train and validation
    plot3d(k + " | Обучающая выборка", 
        pd.concat([x_train, x_test]), 
        pd.concat([y_train, y_test]).values, 
        np.concatenate([y_pred_train, y_pred]), 
        selected_columns, anomaly=False)
    # plot test
    plot3d(k + " | Тестовая выборка", X_test, Y_test.values, Y_pred, selected_columns, anomaly=False)
    print("accuracy train:", clf.score(x_train,y_train), "accuracy validation", accuracy_score(y_test, y_pred), "accuracy_test", accuracy_score(Y_test, Y_pred))
    print(classification_report(y_train, clf.predict(x_train)))    
    print(classification_report(y_test, y_pred))
    print(classification_report(Y_test, Y_pred))


# ### Модели c учителем

# In[ ]:


models = {
    'Логистическая регрессия': LogisticRegression(solver='lbfgs', random_state=0),
    'Метод k-ближайших': KNeighborsClassifier(n_neighbors=3),
    'Случайный лес': RandomForestClassifier(random_state=0),
    'Градиентный бустинг': GradientBoostingClassifier(n_estimators=500, max_depth=2, random_state=0),
    'Мультислойный перцептрон': MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32), alpha=0.001, max_iter=10000),
}

for k,v in models.items():
    print(k)
    report(v, x_train, y_train, x_test, y_test, X_test, Y_test)


# ### Модели детекции аномалий

# In[ ]:


from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

n_samples = len(x_train_anomaly)
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
LOF_novelty = True
# define outlier/anomaly detection methods to be compared
anomaly_algorithms = {
    "Робастная ковариация": EllipticEnvelope(contamination=outliers_fraction),
    "Одноклассовый МОВ": svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1),
    "Изолирующий лес": IsolationForest(contamination=outliers_fraction, random_state=42),
    "Локальный уровень выброса": LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=LOF_novelty)
}

y_train_anomaly = np.full(len(y_train_anomaly.index), 1)
y_test_anomaly = np.where(y_test_anomaly == 1, -1, 1)

for name, algorithm in anomaly_algorithms.items():
        algorithm.fit(x_train_anomaly)

        # fit the data and tag outliers
        if name == "Локальный уровень выброса":
            if not LOF_novelty:
                y_pred = algorithm.fit_predict(x_train_anomaly)
            else:
                y_pred_test = algorithm.predict(x_test_anomaly)
        else:
            y_pred = algorithm.predict(x_train_anomaly)
            y_pred_test = algorithm.predict(x_test_anomaly)
        
        print(name)
        print("*** TRAIN ***")
        print("pred:", y_pred)
        print("train:", y_train_anomaly)
        print("REPORT:", classification_report(y_train_anomaly, y_pred))
        plot3d(name + " | Обучающая выборка", x_train_anomaly, y_train_anomaly, y_pred, selected_columns)

        print("*** TEST ***")
        print("pred:", y_pred_test)
        print("test:", y_test_anomaly)
        print("REPORT:", classification_report(y_test_anomaly, y_pred_test))
        plot3d(name + " | Тестовая выборка ", x_test_anomaly, y_test_anomaly, y_pred_test, selected_columns)
        print()

