import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split

np.random.seed(1)

class FeatureExtractor:
    def __init__(self, train_path, test_path):
        self.df = pd.read_csv(train_path, sep="|")
        self.test = pd.read_csv(test_path, sep="|")
        self.threshold = 1
        
    # удаляем столбцы с маленьким стандартным отклонением 
    def remove_little_sd(self):
        # столбцы с маленьким стандартным отклонением для аномалий
        cols = self.df[self.df['is_insider'] == 1].std()[self.df[self.df['is_insider'] == 1].std() 
            < self.threshold].index.values.tolist()
        # не трогаем таргет столбец
        cols.remove("is_insider")
        self.df.drop(cols, axis=1, inplace=True)
        # столбцы с маленьким стандартным отклонением для нормального поведения
        cols = self.df[self.df['is_insider'] == 0].std()[self.df[self.df['is_insider'] == 0].std() 
            < self.threshold].index.values.tolist()
        # не трогаем таргет столбец
        cols.remove("is_insider")
        self.df.drop(cols, axis=1, inplace=True)
    
    # заполнение пустот в обучаемой выборке с помощью медиан
    # и их использование для заполнения пустот в валидационной выборке
    def fill_with_medians(self):
        medians = {} # словарь для хранения медиан для каждого столбца
       
        for col in self.x_train.columns:
            if col != 'is_insider':
                # медианы         
                medians[col + "_non_insider"] = self.x_train.loc[(self.x_train[col] != 0) 
                                                                 & (self.x_train['is_insider'] == 0), col].median()
                medians[col + "_is_insider"] = self.x_train.loc[(self.x_train[col] != 0) 
                                                                & (self.x_train['is_insider'] == 1), col].median()
                # заполнение пустот в обучаемой выборке с помощью медиан из обучаемой выборки
                self.x_train.loc[(self.x_train[col] == 0) & (self.x_train['is_insider'] == 0), col] \
                    = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0 
                self.x_train.loc[(self.x_train[col] == 0) & (self.x_train['is_insider'] == 1), col] \
                    = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0
                # заполнение пустот в валидационной выборке с помощью медиан из обучаемой выборки
                self.x_test.loc[(self.x_test[col] == 0) & (self.x_test['is_insider'] == 0), col] \
                    = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0
                self.x_test.loc[(self.x_test[col] == 0) & (self.x_test['is_insider'] == 1), col] \
                    = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0
                # заполнение пустот в тестовой выборке с помощью медиан из обучаемой выборки
                self.X_test.loc[(self.X_test[col] == 0) & (self.X_test['is_insider'] == 0), col] \
                    = medians[col + "_non_insider"] if medians[col + "_non_insider"] is not np.nan else 0
                self.X_test.loc[(self.X_test[col] == 0) & (self.X_test['is_insider'] == 1), col] \
                    = medians[col + "_is_insider"] if medians[col + "_is_insider"] is not np.nan else 0

        # очистка памяти
        del medians
    
    # выбор наиболее информативных признаков
    def select_best_features(self):
        transformer = SelectKBest(chi2, k = 3)
        X_new = transformer.fit_transform(self.x_train, self.y_train)

        selected_columns = self.x_train.columns[transformer.get_support()].values

        self.x_train = self.x_train[selected_columns]
        self.x_test = self.x_test[selected_columns]
        self.X_test = self.X_test[selected_columns]

        self.x_train_anomaly = self.x_train_anomaly[selected_columns]
        self.x_test_anomaly = self.x_test_anomaly[selected_columns]

        return selected_columns

    # удаление столбца is_insider из x_train и x_test
    def drop_columns(self):
        self.x_train.drop(columns = ['is_insider'], inplace = True)
        self.x_test.drop(columns = ['is_insider'], inplace = True)
        self.X_test.drop(columns = ['is_insider'], inplace = True)
        self.x_train_anomaly.drop(columns = ['is_insider'], inplace = True)
        self.x_test_anomaly.drop(columns = ['is_insider'], inplace = True)
    
    # разделение данных для классификаторов
    def split_for_clf(self):
        self.y = self.df['is_insider']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.df, self.y, random_state = 0, test_size = 0.3)
        self.X_test = self.test
        self.Y_test = self.test['is_insider']
    
    # разделение данных для детекторов аномалий
    def split_for_anomaly(self):
        self.x_train_anomaly = pd.concat([self.x_train[self.x_train['is_insider'] == 0], 
                                          self.x_test[self.x_test['is_insider'] == 0]]) 
        self.y_train_anomaly = pd.concat([self.y_train[self.y_train == 0], 
                                          self.y_test[self.y_test == 0]])
        self.x_test_anomaly = pd.concat([self.x_train[self.x_train['is_insider'] == 1], self.x_test[self.x_test['is_insider'] == 1], 
            self.X_test[self.X_test['is_insider'] == 1], self.X_test[self.X_test['is_insider'] == 0]])
        self.y_test_anomaly = pd.concat([self.y_train[self.y_train == 1],
            self.y_test[self.y_test == 1], self.Y_test[self.Y_test == 1], self.Y_test[self.Y_test == 0]])
    
    # получение готовых данных
    def get_data(self, clf=True):
        if not clf:
            return self.x_train_anomaly, self.x_test_anomaly, self.y_train_anomaly, self.y_test_anomaly
        return self.x_train, self.x_test, self.y_train, self.y_test, self.X_test, self.Y_test