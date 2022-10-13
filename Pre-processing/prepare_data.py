#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 14:24:26 2022

@author: Cristian Rios
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io.wavfile import read
from scipy.io.wavfile import write
import time

def reduct_frames(names_frames, FPS_base, FPS_target):
    step=np.arange(0, FPS_base, int(FPS_base/7))
    adjust=FPS_base%FPS_target
    frames=np.array(names_frames)[step]
    adjust = len(frames) - FPS_target
    if adjust == 0:
        return frames
    return frames[:-adjust]


def save_video(frames, path_frames, name_user, path_video, name_video, FPS):
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # Be sure to use lower case
    video = cv2.VideoWriter(path_video+name_video, fourcc, FPS,(100,50))
    for frame in frames:
        image = cv2.imread(path_frames+name_user+'/'+frame)
        resize_image = cv2.resize(image, (100,50), interpolation= cv2.INTER_LINEAR)
        video.write(resize_image)
    cv2.destroyAllWindows()
    video.release()

output_video='./data/videos/'
output_audio='./data/audios/'
FPS_target=7
fs_audios=48000

FPS_participants = pd.read_csv("FPS_participants.csv")

output_PD=[]
output_HC=[]

for name in FPS_participants['ids']:
    
    print(name)
    
    if ('PD' in name):
        path_videos='/home/cristian/Cristian/PhD/Databases/FacePark_GITA/lips/readtext/PD/'   
        path_audios='/home/cristian/Cristian/PhD/Databases/FacePark_GITA/audios_readtext/PD/'
    else:
        path_videos='/home/cristian/Cristian/PhD/Databases/FacePark_GITA/lips/readtext/HC/'   
        path_audios='/home/cristian/Cristian/PhD/Databases/FacePark_GITA/audios_readtext/HC/'
    
    name_folder_img = name
    audio_name = name+'.wav'
    
    FPS_user = FPS_participants[FPS_participants["ids"]==name.split(".")[0]]['FPS'].values[0]
    
    file_lips = sorted([img for img in os.listdir(path_videos+name_folder_img) if img.endswith(".jpg")])
    list_int = [int(x[5:-4]) for x in file_lips]
    file_lips=[m for m,_ in sorted(zip(file_lips,list_int), key=lambda pair: pair[1])]
    
    fs, data_audio = read(path_audios+audio_name)
    data_audio = data_audio-np.mean(data_audio)
    data_audio = data_audio/float(np.max(np.abs(data_audio)))
    
    ite=0
    ite_files=0
    for i in range(int(len(data_audio)/fs_audios)):
        name_frames = []
        for j in range(FPS_user):
            temp = file_lips[ite_files]
            if(str(ite+1)+'.jpg' in temp):
                name_frames.append(temp)
                ite_files=ite_files+1
            ite=ite+1
        if(len(name_frames) == FPS_user):
            frame_select = reduct_frames(name_frames, FPS_user, FPS_target)
            #print(len(frame_select))
            
            name_video = name+'_video_'+str(i)+'.avi'
            save_video(frame_select, path_videos, name,output_video, name_video, FPS_target)
            
            frame_audio=data_audio[i*fs_audios : i*fs_audios + fs_audios]
            name_audio = output_audio+name+'_audio_'+str(i)+'.wav'
            write(name_audio, fs_audios , frame_audio)
            time.sleep(0.05)
            
            temp=output_video+name_video+';'+name_audio
            if ('PD' in name):
                output_PD.append(temp)
            else:
                output_HC.append(temp)
        else:
            print("frames incompletos, ite: ", i)
        
        
np.savetxt("file_PD.txt", output_PD,fmt='%s')
np.savetxt("file_HC.txt", output_HC,fmt='%s')
    
    
        
        
    



















