#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:05:33 2022

@author: Cristian Rios
"""

import numpy as np
import pandas as pd
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#%%

drop_parameters=[0,0.2]
lr_parameters=[0.005,0.001,0.0005]
l2_parameters=[0.0001,0.0005,0.001]

ii=0
fold=0
parameters=ut.grid_serch(drop_parameters,lr_parameters,l2_parameters)

for ii in range(len(parameters)):

    print("Dropout: "+str(parameters[ii][0])+"; Learning_rate: "+str(parameters[ii][1])+"; L2_regul: "+str(parameters[ii][2]))
    
    n_kfold=5
    
    name_data = np.loadtxt("name_participants.txt", dtype=str)
    y = np.concatenate((np.ones(30), np.zeros(23)))
    
    acc_total=[]
    acc_bal_total=[]
    sensi_total=[]
    speci_total=[]
    f1_total=[]
    auc_total=[]
    
    y_scores_final=[]
    z_final=[]
    y_final=[]
    id_test_final=[]
    
    index_train_csv = pd.read_csv('k_fold_train.csv')
    index_test_csv = pd.read_csv('k_fold_test.csv')
    
    for jj in range(1,6):
        train = np.array(index_train_csv['fold_'+str(jj)])
        train = (train[~np.isnan(train)]).astype(int)
    
        test = np.array(index_test_csv['fold_'+str(jj)])
        test = (test[~np.isnan(test)]).astype(int)
        
        fold=fold+1
        names_train, name_test = name_data[train], name_data[test]
        y_train, y_test = y[train], y[test]
    
        
        train_dataset, test_dataset = dataset_multimodal(names_train, y_train, name_test, y_test)
        
        train_loader = DataLoader(dataset=train_dataset, batch_size=8,
                                      shuffle=True, num_workers=4, drop_last=True)
        
        validation_loader = DataLoader(dataset=test_dataset, batch_size=8,
                                     shuffle=True, num_workers=4)
        
        print("# Train batchs: ",len(train_loader))
        print("# Test batchs: ",len(validation_loader))
        
        len_batch = len(train_loader)
        
        #%%
        early_stopping = EarlyStopping(patience=15, verbose=True, name="checkpoints/checkpoint"+str(fold)+".pt")
        model_multimodal = multimodal_arquitecture.MultiModalNet(num_classes=2, drop=parameters[ii][0], num_heads=1)
        if torch.cuda.is_available():
            model_multimodal.cuda()
        model_multimodal.train()
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_multimodal.parameters(),lr=parameters[ii][1], weight_decay=parameters[ii][2])
        #%%
        
        loss_aux=[]
        test_loss_aux=[]
        
        num_epochs=50
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
        
        id_test_final.append(ids_test)
        y_scores_final.append(np.array(score_fold))
        z_final.append(z_fold)
        y_final.append(y_fold)
        
        acc=accuracy_score(y_fold,z_fold)
        acc_balan=balanced_accuracy_score(y_fold,z_fold)
        c_m=confusion_matrix(y_fold,z_fold)
        param=precision_recall_fscore_support(y_fold,z_fold, pos_label=1, average='weighted')
        sensi=(c_m[1][1])/(c_m[1][1]+c_m[1][0])
        speci=(c_m[0][0])/(c_m[0][0]+c_m[0][1])
        f1=param[2]
        fpr, tpr, thresholds = metrics.roc_curve(y_fold, score_fold)
        auc=metrics.auc(fpr, tpr)
        
        
        acc_total.append(acc*100)
        acc_bal_total.append(acc_balan*100)
        sensi_total.append(sensi*100)
        speci_total.append(speci*100)
        f1_total.append(f1*100)
        auc_total.append(auc)
        
        ut.plot_results(loss_aux, test_loss_aux, 0, fold,'./results/')
        
            
    print("------------------------ FINAL -------------------------")
    print("Total Accuracy = " + str(np.mean(acc_total)) + " +/- " + str(np.std(acc_total)))
    print("Total Accuracy Balanced = " + str(np.mean(acc_bal_total)) + " +/- " + str(np.std(acc_bal_total)))
    print("Total Sensitivity = " + str(np.mean(sensi_total)) + " +/- " + str(np.std(sensi_total)))
    print("Total Specifiticy = " + str(np.mean(speci_total)) + " +/- " + str(np.std(speci_total)))
    print("Total F1score = " + str(np.mean(f1_total)) + " +/- " + str(np.std(f1_total)))
    print("Total AUC = " + str(np.mean(auc_total)) + " +/- " + str(np.std(auc_total)))
    
    
    ut.save_results('./results/results.csv', 'acc', ii, acc_total)
    ut.save_results('./results/results.csv', 'acc_bal', ii, acc_bal_total)
    ut.save_results('./results/results.csv', 'sensi', ii, sensi_total)
    ut.save_results('./results/results.csv', 'speci', ii, speci_total)
    ut.save_results('./results/results.csv', 'f1-sco', ii, f1_total)
    ut.save_results('./results/results.csv', 'AUC', ii, auc_total)
    
    ut.save_results('./results/results.csv', 'labels', ii, list(np.hstack(y_final)))
    ut.save_results('./results/results.csv', 'predict', ii, list(np.hstack(z_final)))
    ut.save_results('./results/results.csv', 'scores', ii, list(np.hstack(y_scores_final)))
    ut.save_results('./results/results.csv', 'id_test', ii, list(np.hstack(id_test_final)))
    ut.save_results('./results/results.csv', 'parameters', ii, parameters[ii])
    


    
    




















    
    

 
 
 
 
