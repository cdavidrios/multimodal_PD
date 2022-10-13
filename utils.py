#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:27:59 2019

@author: Cristian
"""
import numpy as np
import pandas as pd
import scipy.stats as st
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import roc_curve


#Retorna el accuracy de la red por hablandte
def accuracy_speaker(adress_csv, index_test, accuracy_total, y_true):
    #adress_csv : lugar del archivo csv de la base de datos
    #index_test : posiciones utilizadas para test
    #accuracy_total : list con los acurracies predichos por la red
    #y_true : etiquetas de los hablantes de test
    
    csv_data = pd.read_csv(adress_csv)
    tam=np.array(csv_data.total_imagenes[index_test])
    
    acc=[]
    score=[]
    ini=0
    fin=0
    for i in range(len(tam)):
        if (i!=0):
            fin=fin+1
        ini=fin
        fin=ini + (tam[i])-1
        acc_aux=st.mode(accuracy_total[ini:fin])[0][0]
        acc.append(acc_aux)
        prob=(st.mode(accuracy_total[ini:fin])[1][0])/(len(accuracy_total[ini:fin]))
        
        if(acc_aux==1):
            score.append(prob)
        else:
            score.append(1-prob)
        
        #print(st.mode(accuracy_total[ini:fin])[0][0])
        #print("desde :"+str(ini)+" hasta: "+str(fin)+ " tamaño :"+str(tam[i]))
        
    #accuracy_final=((np.sum(acc==y_true))/(len(y_true)))
    Z=np.array(acc)
    
    accuracy = accuracy_score(y_true,Z)
    c_m = confusion_matrix(y_true,Z)
    fpr, tpr, _ = roc_curve(y_true, np.asarray(score),pos_label=1)
    param = precision_recall_fscore_support(y_true, Z, pos_label=1, average='weighted')
    spec = (c_m[0][0])/(c_m[0][0]+c_m[0][1])
    sensi = (c_m[1][1])/(c_m[1][1]+c_m[1][0])
    f1 = param[2]
    auc = metrics.auc(fpr, tpr)
    
    results=[accuracy, spec, sensi, f1, auc]
    
    return Z, np.asarray(score),results, fpr, tpr


#Guarda en un archivo csv una lista (loss o accuracy) en la ultima fila
def save_results(file_csv, name_result, fold, result):
    #file_csv: nombre.csv del archivo donde quiero guardar el resultado,
    # ejm 'results_spanish.csv'
    #name_result: nombre del resultado a evaluar ejm: loss_train
    #fold: K del fold en el que se llama la funcion ejm: 3
    #result: list a guardar ejm: loss_aux o [40.0]
    result_aux=result
    result_aux.insert(0,name_result)
    result_aux.insert(0,fold)
    with open(file_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(result_aux)
        
        
#Grafica y guarda en la misma figura el accuracy, y las perdidas
def plot_results(loss_train, loss_test,accuracy_speaker, fold, dir_save='./'):
    #accuracy: list del accuracy retornado por la red
    #loss_train: list del loss_train retornado por la red
    #loss_test: list del loss_test retornado por la red
    #fold: K del fold en el que se llama la funcion ejm: 3
    #dir_save: direccion donde se quiere guardar, por defecto './'

    name='results_fold_'+str(fold)
    plt.figure(figsize=(10,6))
    #plt.plot((accuracy),label='Acc_val',color='grey',linewidth=2)
    plt.plot(loss_train,label='loss_train',color='red',linewidth=2)
    plt.plot(loss_test,label='loss_val',color='blue',linewidth=2)
    
    if(accuracy_speaker != 0):
        plt.plot(accuracy_speaker,label='Acc_speaker',color='green',linewidth=2)
    
    plt.legend(fontsize=12)
    plt.title(name ,fontsize=18)
    plt.savefig(dir_save+name+str(".png"))
    #plt.show()
    #plt.close()
    
#Organiza las predicciones de acuerdo a los indices (aleatorios)
def predic_sort(list_pred, list_index):
    #list_pred: lista de las predicciones hechas por la CNN
    #list_index: lista de los indices evaluados
    sort_index = np.argsort(list_index)
    return list_pred[sort_index]

def grid_serch(dropout, learning_rate, l2_regularization):
    
    parameters=[]
    for i in dropout:
        for j in learning_rate:
            for k in l2_regularization:
                aux=[i,j,k]
                parameters.append(aux)
                #print(aux)
                
    return parameters