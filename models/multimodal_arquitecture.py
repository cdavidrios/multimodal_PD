#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 10:13:25 2022

@author: Cristian Rios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer_timm import AttentionBlock, Attention


#%% Video Architecture
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel),
                        )
        self.conv2  = nn.Sequential(
                        nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(outchannel)
                    )
        self.skip = nn.Sequential()
        if stride != 1 or inchannel != self.expansion * outchannel:
            self.skip = nn.Sequential(
                nn.Conv2d(inchannel, self.expansion * outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * outchannel)
            )

    def forward(self, X):
        out = F.relu(self.conv1(X))
        out = self.conv2(out)
        out += self.skip(X)
        out = F.relu(out)
        return out

class VideoCNN(nn.Module):
    def __init__(self, num_classes, drop):
        super(VideoCNN, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride=2)        

        self.fc = nn.Linear(64*ResidualBlock.expansion, num_classes)
        self.dropout = nn.Dropout2d(p=drop)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_classifier(x)
        return x

    def forward_stage1(self, x):
        out = F.relu(self.dropout(self.conv1(x)))
        out = self.dropout(self.layer1(out))
        out = self.dropout(self.layer2(out))
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, (out.size()[2],out.size()[3]))
        out = torch.flatten(out, 1)
        return out

    def forward_classifier(self, x):   
        out = self.fc(x)
        return out


#%% Audio architecture
class AudioCNN(nn.Module):
    def __init__(self, n_input=1, n_output=2, stride=16, n_channel=16, drop=0.3):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=480, stride=12)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(n_channel, 2*n_channel, kernel_size=240, stride=6)
        self.bn2 = nn.BatchNorm1d(2*n_channel)
        self.pool2 = nn.MaxPool1d(2)
        self.lstm1 = torch.nn.LSTM(
            input_size= 32,
            hidden_size=64,
            batch_first=True,
            num_layers=2,
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(9280, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, n_output)

    def forward(self, x):
        x = self.forward_stage1(x)
        x = self.forward_stage2(x)
        x = self.forward_classifier(x)
        return x
    
    def forward_stage1(self, x):
        x = self.dropout(self.conv1(x))
        #print("salida conv1", x.size())
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.dropout(self.conv2(x))
        #print("salida conv2", x.size())
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        #print("salida pooling2", x.size())
        return x
    
    def forward_stage2(self, x):
        x = x.permute(0, 2, 1)
        #print("entrada lstm", x.size())
        x,_ = self.lstm1(x)
        #print("salida lstm", x.size())
        return x

    def forward_classifier(self, x):   
        x = self.flatten(x)
        #print("salida flatten", x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


#%% Multimodal Architecture

class MultiModalNet(nn.Module):
    def __init__(self, num_classes, drop=0, num_heads=1):
        super(MultiModalNet, self).__init__()
        
        self.audio_model = AudioCNN(n_input=1, n_output=num_classes, drop=drop)
        self.visual_model = VideoCNN(num_classes=num_classes, drop=drop)
    
        input_dim_video = 64
        input_dim_audio = 32
        
        self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads)
        self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads)
        
        self.classifier_1 = nn.Sequential(
                nn.Linear(96, num_classes),
            )


    def forward(self, x_audio, x_visual):
        batch_size = x_visual.shape[0]
        num_fps = x_visual.shape[2]
        
        x_audio = self.audio_model.forward_stage1(x_audio)

        #Fusion batch with FPS
        x_visual = x_visual.permute(0,2,1,3,4) 
        x_visual = x_visual.flatten(0,1)
        x_visual = self.visual_model.forward_stage1(x_visual)
        
        x_visual = x_visual.unflatten(0, (batch_size,num_fps))
        x_visual = x_visual.permute(0,2,1)
        
        proj_x_a = x_audio.permute(0,2,1)
        proj_x_v = x_visual.permute(0,2,1)


        _, h_av = self.av1(proj_x_v, proj_x_a)
        _, h_va = self.va1(proj_x_a, proj_x_v)


        if h_av.size(1) > 1: #if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)
        if h_va.size(1) > 1: #if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)
        
        h_av = h_av.sum([-2])
        h_va = h_va.sum([-2])

        x_visual = h_av*x_visual
        x_audio = h_va*x_audio
        
        audio_pooled = x_audio.mean([-1]) #mean accross temporal dimension
        video_pooled = x_visual.mean([-1])
        
        x = torch.cat((audio_pooled, video_pooled), dim=-1)
        x1 = self.classifier_1(x)

        return x1






