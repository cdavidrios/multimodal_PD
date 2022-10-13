#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:05:22 2022

@author: Cristian Rios
"""

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.io.wavfile import read



class ClassDataset(Dataset):
    
    def __init__(self, data):
        self.data=data
        self.len = len(data)
        
    def __getitem__(self, index):
        id_index = self.data[index]['id']
        label_index = self.data[index]['label']
        path_audio = self.data[index]['path_audio']
        path_video = self.data[index]['path_video']
        
        #Load speech signal
        fs, data_audio = read(path_audio)
        data_audio = data_audio-np.mean(data_audio)
        data_audio = data_audio/float(np.max(np.abs(data_audio)))
        
        #Load video
        images = []
        cap = cv2.VideoCapture(path_video)
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            img_normalized = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            images.append(img_normalized)
            
        images = torch.Tensor(np.array(images))
        images = torch.unsqueeze(images, dim=0)
        
        data_audio = torch.Tensor(data_audio)
        data_audio = torch.unsqueeze(data_audio, dim=0)
        
        return id_index, data_audio, images, label_index

    def __len__(self):
        return self.len



def dataset_multimodal(names_train, labels_train, names_test, labels_test):
    annotations_HC = './data/annotations_HC.txt'
    annotations_HC = np.loadtxt(annotations_HC, dtype=str)
    annotations_PD = './data/annotations_PD.txt'
    annotations_PD = np.loadtxt(annotations_PD, dtype=str)
    
    data_train = []
    data_test = []
    for annotation_HC in annotations_HC:
        name_annotation = (annotation_HC.split("/")[3]).split("_video_")[0]
        if (name_annotation in names_train):
            data_train.append({'id': name_annotation,
                          'path_video': annotation_HC.split(";")[0],
                          'path_audio': annotation_HC.split(";")[1],
                          'label':0})
        else:
            data_test.append({'id': name_annotation,
                          'path_video': annotation_HC.split(";")[0],
                          'path_audio': annotation_HC.split(";")[1],
                          'label':0})            
    
    for annotation_PD in annotations_PD:
        name_annotation = (annotation_PD.split("/")[3]).split("_video_")[0]
        if (name_annotation in names_train):
            data_train.append({'id': name_annotation,
                          'path_video': annotation_PD.split(";")[0],
                          'path_audio': annotation_PD.split(";")[1],
                          'label':1})
        else:
            data_test.append({'id': name_annotation,
                          'path_video': annotation_PD.split(";")[0],
                          'path_audio': annotation_PD.split(";")[1],
                          'label':1})     

    data_train = np.array(data_train)
    data_test = np.array(data_test)
    
    dataset_train = ClassDataset(data_train)
    dataset_test = ClassDataset(data_test)

    return dataset_train, dataset_test