import pandas as pd
import numpy as np
import copy
from time import mktime
from datetime import datetime as dt
from features import *


class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.selected_columns = ["то_interval", "ста_interval_first", "ани_interval_first"]
    
    def get_data(self):
        data = self.df[self.selected_columns]

        medians = {}
        for col in self.selected_columns:
            medians[col] = data.loc[data[col] != 0, col].median()
            data.loc[(data[col] == 0), col] = medians[col] if medians[col] is not np.nan else 0
        
        return self.df[self.selected_columns]

class FeatureProcessor:
    def __init__(self):
        self.features = {}
    
    def clear_features(self):
        self.features.clear()
    
    # вычисление признаков для кнопок мыши и униграфов
    def mouse_and_special_keys(self, df, events):
        for event in events:
            if event not in self.features: # в словаре признаков пока не выделена память под эту кнопку
                self.features[event] = copy.deepcopy(spec_features) # выделяем память
            for i, row in df.iterrows():
                if row['action'] == "press" and row['message'] == event:
                    press = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f') # фиксируем время нажатия кнопки
                    # время отпускания нажатой кнопки 
                    # по умолчанию задаётся значением press, 
                    # так как release для последнего события в датасете может отсутствовать
                    release = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                    # время нажатия кнопки после отпускания текущей
                    # по умолчанию задаётся значением press, 
                    # так как press следующей кнопки для последнего события в датасете может отсутствовать
                    press_next = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                    j = 1
                    while True: # ищем время отпускания кнопки
                        try: # пробуем обратиться по индексу i + j
                            next_row = df.iloc[i + j]
                            if next_row['action'] == "release" and next_row['message'] == event:
                                # фиксируем время отпускания текущей кнопки
                                release = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                try: # пробуем обратиться по индексу i +j + 1
                                    # фиксируем время нажатия следующей кнопки
                                    press_next = dt.strptime(df.iloc[i + j + 1]['time'], '%Y-%m-%d %H:%M:%S.%f')
                                except IndexError: # в случае отсутствия такого индекса, т.е. в случае выхода за границу датафрейма
                                    # фиксируем время нажатия следующей кнопки, как время отпускания текущей
                                    press_next = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                finally:
                                    # выходим из вспомогательного цикла в любом случае, так как release был найден
                                    break
                            else:
                                j+=1 # переход к следующей строке
                        except IndexError:
                            # выходим из вспомогательного цикла
                            break
                    # считаем признаки
                    # dwell
                    self.features[event]["dwell"][0] += (release - press).microseconds // 1000 # прибавляем длительность
                    self.features[event]["dwell"][1] += 1 # увеличиваем количество обработанных кнопок event
                    # interval
                    self.features[event]["interval"][0] += (press_next - release).microseconds // 1000 # прибавляем длительность
                    self.features[event]["interval"][1] += 1 # увеличиваем количество обработанных кнопок event
                    # flight
                    self.features[event]["flight"][0] += (press_next - press).microseconds // 1000 # прибавляем длительность
                    self.features[event]["flight"][1] += 1 # увеличиваем количество обработанных кнопок event
                
    # вычисление признаков для диграфов
    def digraph_features(self, df, events):
        for event in events:
            if event not in self.features: # в словаре признаков пока не выделена память под эту кнопку
                self.features[event] = copy.deepcopy(di_features) # выделяем память
            k = 0 # отвечает за индекс текущего символа в диграфе
            first_press = ""
            first_release = ""
            for i, row in df.iterrows():
                if row['action'] == "press":
                    if row['message'][1:-1].lower() == event[k]:
                        press = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f') # фиксируем время нажатия кнопки
                        # время отпускания нажатой кнопки 
                        # по умолчанию задаётся значением press, 
                        # так как release для последнего события в датасете может отсутствовать
                        release = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                        # время нажатия кнопки после отпускания текущей
                        # по умолчанию задаётся значением press, 
                        # так как press следующей кнопки для последнего события в датасете может отсутствовать
                        press_next = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                        j = 1
                        while True: # ищем время отпускания кнопки
                            try: # пробуем обратиться по индексу i + j
                                next_row = df.iloc[i + j]
                                if next_row['action'] == "release" and next_row['message'][1:-1] == event[k]:
                                    # фиксируем время отпускания текущей кнопки
                                    release = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    try: # пробуем обратиться по индексу i +j + 1
                                        # фиксируем время нажатия следующей кнопки
                                        press_next = dt.strptime(df.iloc[i + j + 1]['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    except IndexError: # в случае отсутствия такого индекса, т.е. в случае выхода за границу датафрейма
                                        # фиксируем время нажатия следующей кнопки, как время отпускания текущей
                                        press_next = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    finally:
                                        # выходим из вспомогательного цикла в любом случае, так как release был найден
                                        break
                                else:
                                    j+=1 # переход к следующей строке
                            except IndexError:
                                # выходим из вспомогательного цикла
                                break
                        if k == 0:
                            # сохраняем момент нажатия и отпускания первой клавиши
                            first_press = press
                            first_release = release
                            k+=1
                        else:
                            # считаем признаки
                            # dwell_first
                            self.features[event]["dwell_first"][0] += (first_release - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["dwell_first"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # interval
                            self.features[event]["interval"][0] += (press - first_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["interval"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # flight
                            self.features[event]["flight"][0] += (press - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["flight"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # dwell_second
                            self.features[event]["dwell_second"][0] += (release - press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["dwell_second"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # up_to_up
                            self.features[event]["up_to_up"][0] += (release - first_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["up_to_up"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # latency
                            self.features[event]["latency"][0] += (release - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["latency"][1] += 1 # увеличиваем количество обработанных кнопок event 
                            k = 0
                            first_press = ""
                            first_release = ""
                    else:
                        k = 0
                        first_press = ""
                        first_release = ""
    
    # вычисление признаков для триграфов
    def trigraph_features(self, df, events):
        for event in events:
            if event not in self.features: # в словаре признаков пока не выделена память под эту кнопку
                self.features[event] = copy.deepcopy(tri_features) # выделяем память
            k = 0 # отвечает за индекс текущего символа в диграфе
            first_press = ""
            first_release = ""
            second_press = ""
            seconds_release = ""
            for i, row in df.iterrows():
                if row['action'] == "press":
                    if row['message'][1:-1].lower() == event[k]:
                        press = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f') # фиксируем время нажатия кнопки
                        # время отпускания нажатой кнопки 
                        # по умолчанию задаётся значением press, 
                        # так как release для последнего события в датасете может отсутствовать
                        release = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                        # время нажатия кнопки после отпускания текущей
                        # по умолчанию задаётся значением press, 
                        # так как press следующей кнопки для последнего события в датасете может отсутствовать
                        press_next = dt.strptime(row['time'], '%Y-%m-%d %H:%M:%S.%f')
                        j = 1
                        while True: # ищем время отпускания кнопки
                            try: # пробуем обратиться по индексу i + j
                                next_row = df.iloc[i + j]
                                if next_row['action'] == "release" and next_row['message'][1:-1] == event[k]:
                                    # фиксируем время отпускания текущей кнопки
                                    release = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    try: # пробуем обратиться по индексу i +j + 1
                                        # фиксируем время нажатия следующей кнопки
                                        press_next = dt.strptime(df.iloc[i + j + 1]['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    except IndexError: # в случае отсутствия такого индекса, т.е. в случае выхода за границу датафрейма
                                        # фиксируем время нажатия следующей кнопки, как время отпускания текущей
                                        press_next = dt.strptime(next_row['time'], '%Y-%m-%d %H:%M:%S.%f')
                                    finally:
                                        # выходим из вспомогательного цикла в любом случае, так как release был найден
                                        break
                                else:
                                    j+=1 # переход к следующей строке
                            except IndexError:
                                # выходим из вспомогательного цикла
                                break
                        if k == 0:
                            # сохраняем момент нажатия и отпускания первой клавиши
                            first_press = press
                            first_release = release
                            k+=1
                        elif k == 1:
                            # сохраняем момент нажатия и отпускания первой клавиши
                            second_press = press
                            second_release = release
                            k+=1
                        else:
                            # считаем признаки
                            # dwell_first
                            self.features[event]["dwell_first"][0] += (first_release - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["dwell_first"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # interval_first
                            self.features[event]["interval_first"][0] += (second_press - first_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["interval_first"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # flight_first
                            self.features[event]["flight_first"][0] += (second_press - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["flight_first"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # up_to_up_first
                            self.features[event]["up_to_up_first"][0] += (second_release - first_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["up_to_up_first"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # dwell_second
                            self.features[event]["dwell_second"][0] += (second_release - second_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["dwell_second"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # interval_second
                            self.features[event]["interval_second"][0] += (press - second_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["interval_second"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # flight_second
                            self.features[event]["flight_second"][0] += (press - second_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["flight_second"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # up_to_up_second
                            self.features[event]["up_to_up_second"][0] += (release - second_release).microseconds // 1000 # прибавляем длительность
                            self.features[event]["up_to_up_second"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # dwell_third
                            self.features[event]["dwell_third"][0] += (release - press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["dwell_third"][1] += 1 # увеличиваем количество обработанных кнопок event
                            # latency
                            self.features[event]["latency"][0] += (release - first_press).microseconds // 1000 # прибавляем длительность
                            self.features[event]["latency"][1] += 1 # увеличиваем количество обработанных кнопок event 
                            k = 0
                            first_press = ""
                            first_release = ""
                            second_press = ""
                            second_release = ""
                    else:
                        k = 0
                        first_press = ""
                        first_release = ""
                        second_press = ""
                        second_release = ""
    
    # расчёт средних показателей и запись в файл с признаками
    def calc_average(self, duration):
        averaged = [] # список для хранения усреднённых признаков
        # расчёт средних показателей
        for k, v in self.features.items():
            if k in special_keys: # для специальных признаков кроме временных показателей
                    averaged.append([k, v['dwell'][1] / duration]) # сохраняются ещё и частотные показатели
            for key, value in v.items():
                averaged.append([k + "_" + key, round(value[0] / value[1], 2) if value[1] != 0 else 0])
       
        features = {}
            
        for i in range(len(averaged)):
            features[averaged[i][0]] = str(averaged[i][1])
        return pd.DataFrame([features])
            
