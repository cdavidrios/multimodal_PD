#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:05:33 2022

@author: Cristian Rios
"""

import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics

from models import multimodal_arquitecture
from dataset import dataset_multimodal

import utils as ut
from pytorchtools import EarlyStopping

name_participants = np.loadtxt("name_participants.txt", dtype=str)
y_participants = np.concatenate((np.ones(30), np.zeros(23)))

fold=0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

names_train, name_test, y_train, y_test = train_test_split(name_participants, y_participants, test_size=0.2, random_state=40)


train_dataset, test_dataset = dataset_multimodal(names_train, y_train, name_test, y_test)

train_loader = DataLoader(dataset=train_dataset, batch_size=16,
                              shuffle=True, num_workers=4, drop_last=True)

validation_loader = DataLoader(dataset=test_dataset, batch_size=16,
                             shuffle=True, num_workers=4)

print("# Train batchs: ",len(train_loader))
print("# Test batchs: ",len(validation_loader))

len_batch = len(train_loader)

#%%
early_stopping = EarlyStopping(patience=15, verbose=True, name="checkpoints/checkpoint"+str(fold)+".pt")
model_multimodal = multimodal_arquitecture.MultiModalNet(num_classes=2, drop=0, num_heads=1)
if torch.cuda.is_available():
    model_multimodal.cuda()
model_multimodal.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_multimodal.parameters(),lr=0.01, weight_decay=0.0001)
#%%

loss_aux=[]
test_loss_aux=[]

num_epochs=10
for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    
    for i, (id_name, audio, images, labels) in enumerate(train_loader, 0):
        
        audio = audio.to(device)
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients IMPORTANTE
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model_multimodal(audio, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % len_batch == (len_batch-1):    # print every 2000 mini-batches
            print('[%d, %d, %5d] loss: %.3f' %
                  (fold, epoch + 1, i + 1, running_loss / len_batch))
            loss_aux.append(running_loss/len_batch)
            running_loss = 0.0

            correct = 0
            total = 0
            test_loss = 0
            model_multimodal.eval()
        
            for j, (id_name, audio, images, labels) in enumerate(validation_loader, 0):
                
                audio = audio.to(device)
                images = images.to(device)
                labels = labels.to(device)

                outputs = model_multimodal(audio, images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        
            model_multimodal.train()
            test_loss_aux.append(test_loss/len(validation_loader))
            print('Loss test %.3f, Accuracy test %.3f ' %
                  (test_loss/len(validation_loader),100 * (correct / total)))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            early_stopping(test_loss/len(validation_loader), model_multimodal)

        if early_stopping.early_stop:
            print("-------------Early stopping---------------")
            break
        
    if early_stopping.early_stop:
        print("-------------Early stopping---------------")
        break


#%% Test model

model_multimodal.load_state_dict(torch.load("checkpoints/checkpoint"+str(fold)+".pt"))
model_multimodal.eval()    

total_ids =[]
total_predicted = []
total_labels = []

for j, (id_name, audio, images, labels) in enumerate(validation_loader, 0):
    
    audio = audio.to(device)
    images = images.to(device)
    labels = labels.to(device)

    outputs = model_multimodal(audio, images)
    _, predicted = torch.max(outputs.data, 1)
    
    total_ids.append(np.array(id_name))
    total_predicted.append((predicted.cpu()).numpy())
    total_labels.append((labels.cpu()).numpy())



total_ids = np.hstack(total_ids)
total_predicted = np.hstack(total_predicted)
total_labels = np.hstack(total_labels)

ids_test= list(set(total_ids))

z_fold=[]
y_fold=[]
score_fold=[]

for id_test in ids_test:
    pos_test = np.where(total_ids==id_test)[0]
    pred = total_predicted[pos_test]
    lab = total_labels[pos_test]
    
    if (np.mean(lab)==1.0 or np.mean(lab)==0.0):
        y_fold.append(np.mean(lab))
        z_fold.append(stats.mode(pred)[0][0])
        prob=(stats.mode(pred)[1][0])/(len(pred))
        if(stats.mode(pred)[0][0]==1):
            score_fold.append(prob)
        else:
            score_fold.append(1-prob)
    else:
        print("PROBLEMA CON LOS LABELS")

acc=accuracy_score(y_fold,z_fold)
acc_balan=balanced_accuracy_score(y_fold,z_fold)
c_m=confusion_matrix(y_fold,z_fold)
param=precision_recall_fscore_support(y_fold,z_fold, pos_label=1, average='weighted')
sensi=(c_m[1][1])/(c_m[1][1]+c_m[1][0])
speci=(c_m[0][0])/(c_m[0][0]+c_m[0][1])
f1=param[2]
fpr, tpr, thresholds = metrics.roc_curve(y_fold, score_fold)
auc=metrics.auc(fpr, tpr)


print("Accuracy=",acc*100)
print("Specificity = ", speci*100)
print("Sensivity = ", sensi*100)
print("F1score = ", f1*100)
print("AUC= ", auc)

    

























    
    

 
 
 
 