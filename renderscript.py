#!/usr/bin/python

import sys
import retro
import cv2
from os import listdir
from os.path import isfile, join, isdir, basename, splitext
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc

width = 320
height = 224

width = 320
height = 224

rwidth = 120
rheight = 88

timedelay = 10
FPS = 60 / timedelay
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

def render(file):
    movie = retro.Movie(file)
    file_out = splitext(basename(file))[0]+'.mp4'
    movie.step()
    video = VideoWriter(file_out, fourcc, float(FPS), (rwidth, rheight))
    env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
    env.initial_state = movie.get_state()
    env.reset()
    frame = 0
    framerate = 1
    while movie.step():
        if frame == timedelay:
            video.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            frame = 0
        keys = []
        for i in range(env.num_buttons):
            keys.append(movie.get_key(i, 0))
        _obs, _rew, _done, _info = env.step(keys)
        resized = cv2.resize(_obs, (rwidth, rheight))
        frame += 1
        env.render()
    env.close()
    video.release()
if isdir(sys.argv[1]):
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    onlyfiles.sort()
    for file in onlyfiles:
        if ".bk2" in file :
            print(sys.argv[1])
            print('playing', file)
            print('-----------------------')
            render(file)
else:
    print('playing', sys.argv[1])
    render(sys.argv[1])