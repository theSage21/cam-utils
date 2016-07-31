import os
import numpy as np
import cv2
from subprocess import check_output

def get_lum(frame):
    "Get luminance from the frame"
    R, G, B = frame[:,:,0], frame[:,:,1], frame[:,:,2]
    lum = (0.2126*R + 0.7152*G + 0.0722*B)
    return lum

def add_preference(lum):
    "Add screen brightness preference for given luminance image"
    M, m, me, std = lum.max(), lum.min(), lum.mean(), lum.std()
    brightness = check_output('xbacklight').decode().strip()
    with open('storage/screen_brightness', 'a') as fl:
        fl.write('{},{},{},{},{}\n'.format(M, m, me, std, brightness))

def get_required_brightness(lum):
    "What is the screen brightness preferred for this type of environment"
    with open('storage/screen_brightness', 'r') as fl:
        lines = fl.readlines()
    data = [list(map(float, i.strip().split(','))) for i in lines]
    M, m, me, std, br = ([i[0] for i in data],
            [i[1] for i in data],
            [i[2] for i in data],
            [i[3] for i in data],
            [i[4] for i in data])
    M_, m_, me_, std_ = lum.max(), lum.min(), lum.mean(), lum.std()
    # polyfit all to predict brightness (LinearRegression)
    predictions = []
    for i, j in zip([M, m, me, std], [M_, m_, me_, std_]):
        slope, const = np.polyfit(i, br, deg=1)
        predictions.append(j*slope + const)
    return np.array(predictions).mean()

def set_brightness(value):
    'set brightness value'
    os.system('xbacklight -set {}'.format(value))


def run(frame):
    'entrypoint for each frame'
    lum = get_lum(frame)
    # TODO: a proper way to add preferences
    #add_preference(lum)
    br = get_required_brightness(lum)
    set_brightness(int(float(br)))
