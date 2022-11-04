#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import requests
from bs4 import BeautifulSoup
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tarfile
import urllib
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedShuffleSplit, train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from urllib import request
from zlib import crc32


# In[4]:


from urllib.request import urlopen
from html_table_parser import parser_functions as parser


# In[114]:


data = pd.read_csv("KBO_Profile_12goon_20220217.csv")
data = data[:2885]
data.loc[data["Draft_Type"] != data["Draft_Type"], "Draft_Type"] = 0
idx = data[data['Draft_Type']=='자유선발'].index
idy = data[data['Draft_Type']=='해외파 특별지명'].index
data.drop(idx, inplace=True)
data.drop(idy, inplace=True)
data.dropna(subset = ["Born_Year"])
data.dropna(subset = ["Born_Month"])
data.dropna(subset = ["Born_Day"])


# In[115]:


data["Born_Year"].astype(float)
data["Born_Month"].astype(float)
data["Born_Day"].astype(float)


# In[116]:


data['Born_Year'] = pd.to_numeric(data['Born_Year'], errors='coerce').fillna(0).astype(int)
data['Born_Month'] = pd.to_numeric(data['Born_Month'], errors='coerce').fillna(0).astype(int)
data['Born_Day'] = pd.to_numeric(data['Born_Day'], errors='coerce').fillna(0).astype(int)


# In[117]:


data['Birth_Date'] = data['Born_Year'].astype(str) + "-" + data['Born_Month'].astype(str) + "-" + data['Born_Day'].astype(str)


# In[118]:


data.dropna(subset = ["Debut_Year"])


# In[119]:


data00 = data.loc[:, ["Name", "Birth_Date", "univ", "Draft_Team"]]

kbo_high = data00[data00["univ"].isnull()]
len_kbo_high = len(kbo_high)


# In[120]:


data01 = data.loc[:, ["Name", "Birth_Date", "univ", "Draft_Team"]]

kbo_high01 = data01[data01["univ"].isnull()]
len_kbo_high01 = len(kbo_high01)


# In[121]:


kbo_high = data00[data00["univ"].isnull()]

kbo_high_player = kbo_high.reset_index()
len_kbo_high_player = len(kbo_high_player)

kbo_high_player["Team"] = "NaN"
kbo_high_player["G"] = 0
kbo_high_player["WAR"] = 0


# In[38]:


import collections
collections.Callable = collections.abc.Callable


# In[176]:


condition_Heroes = (data.Draft_Team == '삼미') | (data.Draft_Team == '청보') | (data.Draft_Team == '태평양') | (data.Draft_Team ==  '현대') | (data.Draft_Team == '히어로즈') | (data.Draft_Team == '넥센') | (data.Draft_Team == '키움')
Heroes = data[condition_Heroes]

Heroes_player = Heroes.loc[:, ["Name", "Birth_Date"]]
Heroes_player = Heroes_player.reset_index()

Heroes_player["G"] = 0
Heroes_player["WAR"] = 0
Heroes_player["포지션"] = "NaN"
for i in range(8):
    Heroes_player["G_" + str(i+1)] = 0
    Heroes_player["WAR_" + str(i+1)] = 0

len_Heroes = len(Heroes_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Heroes-1):
    name = urllib.parse.quote(Heroes_player.iloc[i][1])
    birth = Heroes_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Heroes_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Heroes_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Heroes_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Heroes_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Heroes_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Heroes_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Heroes_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Heroes_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Heroes_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Heroes_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Heroes_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Heroes_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Heroes_player["포지션"][i] = "C"
            else:
                Heroes_player["포지션"][i] = "NaN"
            
        Heroes_player["G"][i] = g
        Heroes_player["WAR"][i] = war


# In[177]:


Heroes_player.to_csv('Heroes.csv', encoding='cp949')


# In[174]:


condition_Wyverns = (data.Draft_Team == 'SK') | (data.Draft_Team == '쌍방울') | (data.Draft_Team == 'SSG')
Wyverns = data[condition_Wyverns]

Wyverns_player = Wyverns.loc[:, ["Name", "Birth_Date"]]
Wyverns_player = Wyverns_player.reset_index()

Wyverns_player["G"] = 0
Wyverns_player["WAR"] = 0
Wyverns_player["포지션"] = "NaN"
for i in range(8):
    Wyverns_player["G_" + str(i+1)] = 0
    Wyverns_player["WAR_" + str(i+1)] = 0

len_Wyverns = len(Wyverns_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Wyverns-1):
    name = urllib.parse.quote(Wyverns_player.iloc[i][1])
    birth = Wyverns_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Wyverns_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Wyverns_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Wyverns_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Wyverns_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Wyverns_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Wyverns_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Wyverns_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Wyverns_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Wyverns_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Wyverns_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Wyverns_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Wyverns_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Wyverns_player["포지션"][i] = "C"
            else:
                Wyverns_player["포지션"][i] = "NaN"
            
        Wyverns_player["G"][i] = g
        Wyverns_player["WAR"][i] = war


# In[175]:


Wyverns_player.to_csv('Wyverns.csv', encoding='cp949')


# In[172]:


condition_Wiz = (data.Draft_Team == 'KT')
Wiz = data[condition_Wiz]

Wiz_player = Wiz.loc[:, ["Name", "Birth_Date"]]
Wiz_player = Wiz_player.reset_index()

Wiz_player["G"] = 0
Wiz_player["WAR"] = 0
Wiz_player["포지션"] = "NaN"
for i in range(8):
    Wiz_player["G_" + str(i+1)] = 0
    Wiz_player["WAR_" + str(i+1)] = 0

len_Wiz = len(Wiz_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Wiz-1):
    name = urllib.parse.quote(Wiz_player.iloc[i][1])
    birth = Wiz_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Wiz_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Wiz_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Wiz_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Wiz_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Wiz_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Wiz_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Wiz_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Wiz_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Wiz_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Wiz_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Wiz_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Wiz_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Wiz_player["포지션"][i] = "C"
            else:
                Wiz_player["포지션"][i] = "NaN"
            
        Wiz_player["G"][i] = g
        Wiz_player["WAR"][i] = war


# In[173]:


Wiz_player.to_csv('Wiz.csv', encoding='cp949')


# In[170]:


condition_Dinos = (data.Draft_Team == 'NC')
Dinos = data[condition_Dinos]

Dinos_player = Dinos.loc[:, ["Name", "Birth_Date"]]
Dinos_player = Dinos_player.reset_index()

Dinos_player["G"] = 0
Dinos_player["WAR"] = 0
Dinos_player["포지션"] = "NaN"
for i in range(8):
    Dinos_player["G_" + str(i+1)] = 0
    Dinos_player["WAR_" + str(i+1)] = 0

len_Dinos = len(Dinos_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Dinos-1):
    name = urllib.parse.quote(Dinos_player.iloc[i][1])
    birth = Dinos_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Dinos_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Dinos_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Dinos_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Dinos_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Dinos_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Dinos_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Dinos_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Dinos_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Dinos_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Dinos_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Dinos_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Dinos_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Dinos_player["포지션"][i] = "C"
            else:
                Dinos_player["포지션"][i] = "NaN"
            
        Dinos_player["G"][i] = g
        Dinos_player["WAR"][i] = war


# In[171]:


Dinos_player.to_csv('Dinos.csv', encoding='cp949')


# In[168]:


condition_Twins = (data.Draft_Team == 'LG') | (data.Draft_Team == 'MBC')
Twins = data[condition_Twins]

Twins_player = Twins.loc[:, ["Name", "Birth_Date"]]
Twins_player = Twins_player.reset_index()

Twins_player["G"] = 0
Twins_player["WAR"] = 0
Twins_player["포지션"] = "NaN"
for i in range(8):
    Twins_player["G_" + str(i+1)] = 0
    Twins_player["WAR_" + str(i+1)] = 0

len_Twins = len(Twins_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Twins-1):
    name = urllib.parse.quote(Twins_player.iloc[i][1])
    birth = Twins_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Twins_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Twins_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Twins_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Twins_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Twins_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Twins_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Twins_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Twins_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Twins_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Twins_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Twins_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Twins_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Twins_player["포지션"][i] = "C"
            else:
                Twins_player["포지션"][i] = "NaN"
            
        Twins_player["G"][i] = g
        Twins_player["WAR"][i] = war


# In[169]:


Twins_player.to_csv('Twins.csv', encoding='cp949')


# In[166]:


condition_Tigers = (data.Draft_Team == '해태') | (data.Draft_Team == 'KIA')
Tigers = data[condition_Tigers]

Tigers_player = Tigers.loc[:, ["Name", "Birth_Date"]]
Tigers_player = Tigers_player.reset_index()

Tigers_player["G"] = 0
Tigers_player["WAR"] = 0
Tigers_player["포지션"] = "NaN"
for i in range(8):
    Tigers_player["G_" + str(i+1)] = 0
    Tigers_player["WAR_" + str(i+1)] = 0

len_Tigers = len(Tigers_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Tigers-1):
    name = urllib.parse.quote(Tigers_player.iloc[i][1])
    birth = Tigers_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Tigers_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Tigers_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Tigers_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Tigers_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Tigers_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Tigers_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Tigers_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Tigers_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Tigers_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Tigers_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Tigers_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Tigers_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Tigers_player["포지션"][i] = "C"
            else:
                Tigers_player["포지션"][i] = "NaN"
            
        Tigers_player["G"][i] = g
        Tigers_player["WAR"][i] = war


# In[167]:


Tigers_player.to_csv('Tigers.csv', encoding='cp949')


# In[164]:


condition_Eagles = (data.Draft_Team == '빙그레') | (data.Draft_Team == '한화')
Eagles = data[condition_Eagles]

Eagles_player = Eagles.loc[:, ["Name", "Birth_Date"]]
Eagles_player = Eagles_player.reset_index()

Eagles_player["G"] = 0
Eagles_player["WAR"] = 0
Eagles_player["포지션"] = "NaN"
for i in range(8):
    Eagles_player["G_" + str(i+1)] = 0
    Eagles_player["WAR_" + str(i+1)] = 0

len_Eagles = len(Eagles_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Eagles-1):
    name = urllib.parse.quote(Eagles_player.iloc[i][1])
    birth = Eagles_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Eagles_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Eagles_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Eagles_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Eagles_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Eagles_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Eagles_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Eagles_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Eagles_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Eagles_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Eagles_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Eagles_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Eagles_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Eagles_player["포지션"][i] = "C"
            else:
                Eagles_player["포지션"][i] = "NaN"
            
        Eagles_player["G"][i] = g
        Eagles_player["WAR"][i] = war


# In[165]:


Eagles_player.to_csv('Eagles.csv', encoding='cp949')


# In[162]:


condition_Giants = (data.Draft_Team == '롯데')
Giants = data[condition_Giants]

Giants_player = Giants.loc[:, ["Name", "Birth_Date"]]
Giants_player = Giants_player.reset_index()

Giants_player["G"] = 0
Giants_player["WAR"] = 0
Giants_player["포지션"] = "NaN"
for i in range(8):
    Giants_player["G_" + str(i+1)] = 0
    Giants_player["WAR_" + str(i+1)] = 0

len_Giants = len(Giants_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Giants-1):
    name = urllib.parse.quote(Giants_player.iloc[i][1])
    birth = Giants_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Giants_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Giants_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Giants_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Giants_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Giants_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Giants_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Giants_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Giants_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Giants_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Giants_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Giants_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Giants_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Giants_player["포지션"][i] = "C"
            else:
                Giants_player["포지션"][i] = "NaN"
            
        Giants_player["G"][i] = g
        Giants_player["WAR"][i] = war


# In[163]:


Giants_player.to_csv('Giants.csv', encoding='cp949')


# In[160]:


condition_Lions = (data.Draft_Team == '삼성')
Lions = data[condition_Lions]

Lions_player = Lions.loc[:, ["Name", "Birth_Date"]]
Lions_player = Lions_player.reset_index()

Lions_player["G"] = 0
Lions_player["WAR"] = 0
Lions_player["포지션"] = "NaN"
for i in range(8):
    Lions_player["G_" + str(i+1)] = 0
    Lions_player["WAR_" + str(i+1)] = 0

len_Lions = len(Lions_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Lions-1):
    name = urllib.parse.quote(Lions_player.iloc[i][1])
    birth = Lions_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Lions_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Lions_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Lions_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Lions_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Lions_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Lions_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Lions_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Lions_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Lions_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Lions_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Lions_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Lions_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Lions_player["포지션"][i] = "C"
            else:
                Lions_player["포지션"][i] = "NaN"
            
        Lions_player["G"][i] = g
        Lions_player["WAR"][i] = war


# In[161]:


Lions_player.to_csv('Lions.csv', encoding='cp949')


# In[158]:


condition_Bears = (data.Draft_Team == '두산') | (data.Draft_Team == 'OB')
Bears = data[condition_Bears]

Bears_player = Bears.loc[:, ["Name", "Birth_Date"]]
Bears_player = Bears_player.reset_index()

Bears_player["G"] = 0
Bears_player["WAR"] = 0
Bears_player["포지션"] = "NaN"
for i in range(8):
    Bears_player["G_" + str(i+1)] = 0
    Bears_player["WAR_" + str(i+1)] = 0

len_Bears = len(Bears_player)

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_Bears-1):
    name = urllib.parse.quote(Bears_player.iloc[i][1])
    birth = Bears_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    Bears_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    Bears_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    Bears_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    Bears_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            Bears_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    Bears_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    Bears_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    Bears_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    Bears_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                Bears_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                Bears_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                Bears_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                Bears_player["포지션"][i] = "C"
            else:
                Bears_player["포지션"][i] = "NaN"
            
        Bears_player["G"][i] = g
        Bears_player["WAR"][i] = war


# In[159]:


Bears_player.to_csv('Bears.csv', encoding='cp949')


# In[156]:


kbo_univ = data00[data00["univ"].notnull()]

kbo_univ_player = kbo_univ.reset_index()
len_kbo_univ_player = len(kbo_univ_player)

kbo_univ_player["Team"] = "NaN"
kbo_univ_player["G"] = 0
kbo_univ_player["WAR"] = 0
kbo_univ_player["포지션"] = "NaN"

for i in range(8):
    kbo_univ_player["G_" + str(i+1)] = 0
    kbo_univ_player["WAR_" + str(i+1)] = 0

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_kbo_univ_player-1):
    if (kbo_univ.iloc[i][3] == "OB") | (kbo_univ.iloc[i][3] == "두산"):
        kbo_univ_player["Team"][i] = "Bears"
    elif (kbo_univ.iloc[i][3] == "삼성"):
        kbo_univ_player["Team"][i] = "Lions"
    elif (kbo_univ.iloc[i][3] == "롯데"):
        kbo_univ_player["Team"][i] = "Jiants"
    elif (kbo_univ.iloc[i][3] == "빙그레") | (kbo_univ.iloc[i][3] == "한화"):
        kbo_univ_player["Team"][i] = "Eagels"
    elif (kbo_univ.iloc[i][3] == "해태") | (kbo_univ.iloc[i][3] == "KIA"):
        kbo_univ_player["Team"][i] = "Tigers"
    elif (kbo_univ.iloc[i][3] == "LG"):
        kbo_univ_player["Team"][i] = "Twins"
    elif (kbo_univ.iloc[i][3] == "NC"):
        kbo_univ_player["Team"][i] = "Dinos"
    elif (kbo_univ.iloc[i][3] == "KT"):
        kbo_univ_player["Team"][i] = "Wiz"
    elif (kbo_univ.iloc[i][3] == "SK") | (kbo_univ.iloc[i][3] == "쌍방울") | (kbo_univ.iloc[i][3] == "SSG"):
        kbo_univ_player["Team"][i] = "Wyverns"
    elif (kbo_univ.iloc[i][3] == "삼미") | (kbo_univ.iloc[i][3] == "청보") | (kbo_univ.iloc[i][3] == "태평양") | (kbo_univ.iloc[i][3] == "현대") | (kbo_univ.iloc[i][3] == "히어로즈") | (kbo_univ.iloc[i][3] == "넥센") | (kbo_univ.iloc[i][3] == "키움"):
        kbo_univ_player["Team"][i] = "Heroes"
    else:
        kbo_univ_player.iloc[i][3] = "NaN"
    name = urllib.parse.quote(kbo_univ_player.iloc[i][1])
    birth = kbo_univ_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    kbo_univ_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    kbo_univ_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    kbo_univ_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    kbo_univ_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            kbo_univ_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    kbo_univ_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    kbo_univ_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    kbo_univ_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    kbo_univ_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                kbo_univ_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                kbo_univ_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                kbo_univ_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                kbo_univ_player["포지션"][i] = "C"
            else:
                kbo_univ_player["포지션"][i] = "NaN"
            
        kbo_univ_player["G"][i] = g
        kbo_univ_player["WAR"][i] = war


# In[157]:


kbo_univ_player.to_csv('kbo_univ.csv', encoding='cp949')


# In[152]:


kbo_high = data00[data00["univ"].isnull()]

kbo_high_player = kbo_high.reset_index()
len_kbo_high_player = len(kbo_high_player)

kbo_high_player["Team"] = "NaN"
kbo_high_player["G"] = 0
kbo_high_player["WAR"] = 0
kbo_high_player["포지션"] = "NaN"

for i in range(8):
    kbo_high_player["G_" + str(i+1)] = 0
    kbo_high_player["WAR_" + str(i+1)] = 0

url0 = "http://www.statiz.co.kr/player.php?opt=1&name="
for i in range(0, len_kbo_high_player-1):
    if (kbo_high.iloc[i][3] == "OB") | (kbo_high.iloc[i][3] == "두산"):
        kbo_high_player["Team"][i] = "Bears"
    elif (kbo_high.iloc[i][3] == "삼성"):
        kbo_high_player["Team"][i] = "Lions"
    elif (kbo_high.iloc[i][3] == "롯데"):
        kbo_high_player["Team"][i] = "Jiants"
    elif (kbo_high.iloc[i][3] == "빙그레") | (kbo_high.iloc[i][3] == "한화"):
        kbo_high_player["Team"][i] = "Eagels"
    elif (kbo_high.iloc[i][3] == "해태") | (kbo_high.iloc[i][3] == "KIA"):
        kbo_high_player["Team"][i] = "Tigers"
    elif (kbo_high.iloc[i][3] == "LG"):
        kbo_high_player["Team"][i] = "Twins"
    elif (kbo_high.iloc[i][3] == "NC"):
        kbo_high_player["Team"][i] = "Dinos"
    elif (kbo_high.iloc[i][3] == "KT"):
        kbo_high_player["Team"][i] = "Wiz"
    elif (kbo_high.iloc[i][3] == "SK") | (kbo_high.iloc[i][3] == "쌍방울") | (kbo_high.iloc[i][3] == "SSG"):
        kbo_high_player["Team"][i] = "Wyverns"
    elif (kbo_high.iloc[i][3] == "삼미") | (kbo_high.iloc[i][3] == "청보") | (kbo_high.iloc[i][3] == "태평양") | (kbo_high.iloc[i][3] == "현대") | (kbo_high.iloc[i][3] == "히어로즈") | (kbo_high.iloc[i][3] == "넥센") | (kbo_high.iloc[i][3] == "키움"):
        kbo_high_player["Team"][i] = "Heroes"
    else:
        kbo_high_player.iloc[i][5] = "NaN"
    name = urllib.parse.quote(kbo_high_player.iloc[i][1])
    birth = kbo_high_player.iloc[i][2]
    url = url0 + name + "&birth=" + birth
    
    result = urlopen(url)
    html = result.read()
    soup = BeautifulSoup(html, 'html.parser')
    temp = soup.find_all('table')
    
    if len(temp) != 3:
        continue
    else:
        if len(temp[2]) < 4:
            url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=0"
            result = urlopen(url)
            html = result.read()
            soup = BeautifulSoup(html, 'html.parser')
            temp = soup.find_all('table')
            if len(temp[2]) < 4:
                url = "http://www.statiz.co.kr/player.php?opt=1&sopt=0&name=" + name + "&birth=" + birth +"&re=1"
                result = urlopen(url)
                html = result.read()
                soup = BeautifulSoup(html, 'html.parser')
                temp = soup.find_all('table')
                
        table = pd.read_html(url, header=0, encoding='utf-8')
        table = table[1]
        table = pd.DataFrame(table)
            
        if ("WAR" in table.columns.tolist()):
            table = table.loc[:,["출 장", "WAR"]][:-1]
            id1 = table[table["WAR"]=='WAR'].index
            table.drop(id1, inplace=True)
            id2 = table[table["WAR"]==''].index
            table.drop(id2, inplace=True)
            id3 = table[table["출 장"]=='출 장'].index
            table.drop(id3, inplace=True)
            id4 = table[table["출 장"]==''].index
            table.drop(id4, inplace=True)
            table = table.astype(float)
            
            if (len(table) > 8):
                for j in range(8):
                    kbo_high_player["G_" + str(j+1)][i] = table.iloc[j][0]
                    kbo_high_player["WAR_" + str(j+1)][i] = table.iloc[j][1]
                g = round(sum(table["출 장"][:8]), 2)
                war = round(sum(table["WAR"][:8]), 2)
            else:
                for k in range(len(table)):
                    kbo_high_player["G_" + str(k+1)][i] = table.iloc[k][0]
                    kbo_high_player["WAR_" + str(k+1)][i] = table.iloc[k][1]
                g = round(sum(table["출 장"]), 2)
                war = round(sum(table["WAR"]), 2)
            kbo_high_player["포지션"][i] = "P"
        else:
            table0 = table.loc[:, ["G", "WAR*"]][:-1]
            id1 = table0[table0["WAR*"]=='WAR*'].index
            table0.drop(id1, inplace=True)
            id2 = table0[table0["WAR*"]==''].index
            table0.drop(id2, inplace=True)
            id3 = table0[table0["G"]=='G'].index
            table0.drop(id3, inplace=True)
            id4 = table0[table0["G"]==''].index
            table0.drop(id4, inplace=True)
            table0 = table0.astype(float)
            
            if (len(table0) > 8):
                for j in range(8):
                    kbo_high_player["G_" + str(j+1)][i] = table0.iloc[j][0]
                    kbo_high_player["WAR_" + str(j+1)][i] = table0.iloc[j][1]
                g = round(sum(table0["G"][:8]), 2)
                war = round(sum(table0["WAR*"][:8]), 2)
            else:
                for k in range(len(table0)):
                    kbo_high_player["G_" + str(k+1)][i] = table0.iloc[k][0]
                    kbo_high_player["WAR_" + str(k+1)][i] = table0.iloc[k][1]
                g = round(sum(table0["G"]), 2)
                war = round(sum(table0["WAR*"]), 2)
            if (table.iloc[-1][3] == "LF") | (table.iloc[-1][3] == "CF") | (table.iloc[-1][3] == "RF"):
                kbo_high_player["포지션"][i] = "OF"
            elif (table.iloc[-1][3] == "2B") | (table.iloc[-1][3] == "3B") | (table.iloc[-1][3] == "SS"):
                kbo_high_player["포지션"][i] = "IF"
            elif (table.iloc[-1][3] == "1B") | (table.iloc[-1][3] == "DH"):
                kbo_high_player["포지션"][i] = "1B/DH"
            elif (table.iloc[-1][3] == "C"):
                kbo_high_player["포지션"][i] = "C"
            else:
                kbo_high_player["포지션"][i] = "NaN"
        kbo_high_player["G"][i] = g

        kbo_high_player["WAR"][i] = war


# In[154]:


kbo_high_player.to_csv('kbo_high.csv', encoding='cp949')

