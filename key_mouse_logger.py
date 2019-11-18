from pynput import keyboard
from pynput import mouse
import logging

formatter = logging.Formatter('%(asctime)s|%(message)s')
path = "./"
listeners_started = False
record_events = False

def setup_logger(name, log_file, level=logging.INFO):
    # remove all old handlers
    logger = logging.getLogger(name)
    for hdlr in logger.handlers[:]: 
        logger.removeHandler(hdlr)

    # create new handler
    handler = logging.FileHandler(path + log_file)        
    handler.setFormatter(formatter)

    # connect handler to logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

keyboard_logger = setup_logger("keyboard", "keyboard.csv")
mouse_logger = setup_logger("mouse", "mouse.csv")

# keyboard key press
def on_press(key):
    if record_events:
        keyboard_logger.info("press|{0}".format(key))

# keyboard press release
def on_release(key):
    if record_events:
        keyboard_logger.info("release|{0}".format(key))

# mouse keys press and release
def on_click(x, y, button, pressed):
    if record_events:
        if pressed:
            mouse_logger.info("press|{0}".format(button))
        else:
            mouse_logger.info("release|{0}".format(button))

# def on_move(x, y):
#     print('Pointer moved to {0}'.format(
#         (x, y)))

# def on_scroll(x, y, dx, dy):
#     print('Scrolled {0} at {1}'.format(
#         'down' if dy < 0 else 'up',
#         (x, y)))

keyboard_listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)

mouse_listener = mouse.Listener(
    on_click=on_click)

def main():
    global listeners_started, record_events
    # information about participant
    name = input("Please, enter your name: ")
    age = input("Please, enter you age: ")
    gender = input("Please, enter your gender, m - male, f - female: ")

    # write participant information into file
    with open(path + "participants.csv", "a") as f: 
        f.write(name + "|" + age + "|" + gender + "\n")

    # scenarios menu
    while True:
        print("1) Non-insider scenario\n2) Insider scenario\n3) Exit")
        choice = int(input("Your choice: "))
        if choice == 3:
            break
        else:
            keyboard_logger_filename = "keyboard_"
            mouse_logger_filename = "mouse_"
            if choice == 1:
                scenario = input("Enter non-insider scenario number: ")
                keyboard_logger_filename += name + "_non_insider_" + scenario + ".csv"
                mouse_logger_filename += name + "_non_insider_" + scenario + ".csv"
            else:
                scenario = input("Enter insider scenario number: ")
                keyboard_logger_filename += name + "_insider_" + scenario + ".csv"  
                mouse_logger_filename += name + "_insider_" + scenario + ".csv"
            # keyboard file logger
            setup_logger("keyboard", keyboard_logger_filename)
            # mouse file logger
            setup_logger("mouse", mouse_logger_filename)
            # start logging
            record_events = True
            # event listeners should be started just once
            if listeners_started == False:
                keyboard_listener.start()
                mouse_listener.start()
                listeners_started = True
            stop_scenario = input("Enter S to stop scenario: ")
            if stop_scenario == "S": # stop logging and go to scenarios menu
                record_events = False

main()