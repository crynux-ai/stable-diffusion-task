import datetime


def log(msg):
    print("[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "] [SDTask] " + msg)
