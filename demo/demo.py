import logging
import time, threading
import pandas as pd
import pynput
from pynput import keyboard
from processor import FeatureProcessor, FeatureExtractor
from joblib import load
from time import mktime
from datetime import datetime as dt
from features import *
from pyfcm import FCMNotification

import warnings
warnings.filterwarnings('ignore')

record_events = True
model = load("isolation_forest.joblib")
keyboard_data = []

# keyboard key press
def on_press(key):
    global record_events, keyboard_data
    if record_events:
        keyboard_data.append({'time': str(dt.now()), "action": "press", "message": "{0}".format(key)})

# keyboard press release
def on_release(key):
    global record_events, keyboard_data
    if record_events:
        keyboard_data.append({'time': str(dt.now()), "action": "release", "message": "{0}".format(key)})

def start_timer():
    global record_events, keyboard_data
    record_events = False

    fp = FeatureProcessor()
    
    if len(keyboard_data) > 10:
        kb = pd.DataFrame(keyboard_data)
        keyboard_data.clear()

        # расчёт времени выполнения сценария в минутах
        start = mktime(dt.strptime(kb.iloc[0][0], '%Y-%m-%d %H:%M:%S.%f').timetuple())
        end = mktime(dt.strptime(kb.iloc[-1][0], '%Y-%m-%d %H:%M:%S.%f').timetuple())
        duration = (end - start) / 60

        # подсчёт признаков
        fp.mouse_and_special_keys(kb, special_keys)
        fp.digraph_features(kb, ru_di)
        fp.trigraph_features(kb, ru_tri)
        features = fp.calc_average(duration)

        # выделение итогового вектора признаков
        fe = FeatureExtractor(features)
        data = fe.get_data()

        res = model.predict(data)

        if res[0] == -1:
            print("Anomaly detected!")
            push_service = FCMNotification(api_key="AAAAwFoVf_4:APA91bGUmrwFJyrL4yt1WlK178VjELYaicPBK4wqswfJQmROHxP0DXPr9vHPDSDAUa_pctcgzuXXvCqCB252q9VexEzkIPVx17rQNwmerJA8_SyXLRqlJFmufXPtt8uMbN4jyMCXTlxZ")
            registration_id = "d5RjZUobQUW3uGsoEnw2pX:APA91bEfVgua18qqXrhDonN6VUjANiOrRiE_2QzEZT9oH94YvYZm0D1VGm0AxHMw8aflOTmjru60M076FS_7h20e0YuzorE7ZCQRwfpLRIycZCO_1i4KClCmt3LUqnyGLCgQtVrGlZuX"
            message_title = "ВНИМАНИЕ!"
            message_body = "Обнаружена аномальная активность!"
            extra_notification_kwargs = {
                "sound": "default"
            }
            result = push_service.notify_single_device(
                registration_id=registration_id, 
                message_title=message_title, 
                message_body=message_body,
                extra_notification_kwargs=extra_notification_kwargs)
        
    threading.Timer(10, start_timer).start()
    record_events = True

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

def main():
    global record_events
    listeners_started = False
    
    while True:
        try:
            if listeners_started == False:
                # event listeners should be started just once
                keyboard_listener.start()
                start_timer()
                listeners_started = True
        except KeyboardInterrupt:
            break  

if __name__ == "__main__":
    main()