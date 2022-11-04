#!/usr/bin/env python
# coding: utf-8

# In[100]:


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


# In[2]:


url="https://www.fangraphs.com/leaders.aspx?pos=all&stats=bat&lg=all&qual=y&type=8&season="


# In[3]:


filename = "battingstats.csv"
f = open(filename, "w", encoding="utf8", newline="")
writer = csv.writer(f)

title = "#	Name	Team	G	PA	HR	R	RBI	SB	BB%	K%	ISO	BABIP	AVG	OBP	SLG	wOBA	xwOBA	wRC+	BsR	Off	Def	WAR".split("\t")
print(type(title))
writer.writerow(title)

for year in range(2010,2020):
    for page in range(1,4):
        res = requests.get(url + str(year) +"&month=0&season1=" + str(year) + "&ind=0&team=0&rost=0&age=0&filter=&players=0&page=" + str(page) + "_50")
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        
        data_rows = soup.find("table", attrs={"class":"rgMasterTable"}).find("tbody").find_all("tr")
        for row in data_rows:
            columns = row.find_all("td")
            data = [column.get_text().strip() for column in columns]
            #print(data)
            writer.writerow(data)


# In[4]:


stat = pd.read_csv('battingstats.csv')
stat


# In[5]:


stat.info()


# In[6]:


stat["WAR"].value_counts()


# In[7]:


stat.describe()


# In[8]:


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[9]:


train_set, test_set = split_train_test(stat, 0.2)
len(test_set) / (len(train_set) + len(test_set))


# In[10]:


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


# In[11]:


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# In[12]:


stat_with_id = stat.reset_index()


# In[13]:


train_set, test_set = split_train_test_by_id(stat_with_id, 0.2, "index")
len(test_set) / (len(train_set) + len(test_set))


# In[14]:


train_set, test_set = train_test_split(stat, test_size=0.2, random_state=42)
len(test_set) / (len(train_set) + len(test_set))


# In[15]:


stat["wRC+"].hist()


# In[16]:


stat["wRC+_cat"] = pd.cut(stat["wRC+"],
                          bins = [0, 80, 100, 120, 140, np.inf],
                          labels = [1, 2, 3, 4, 5])


# In[17]:


stat["wRC+_cat"].value_counts()


# In[18]:


stat["wRC+_cat"].hist()


# In[19]:


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=50)
for train_index, test_index in split.split(stat, stat["wRC+_cat"]):
    train_set = stat.loc[train_index]
    test_set = stat.loc[test_index]


# In[20]:


len(test_set) / (len(train_set) + len(test_set))


# In[21]:


train_set["wRC+_cat"].value_counts() / len(train_set)


# In[22]:


test_set["wRC+_cat"].value_counts() / len(test_set)


# In[23]:


for set_ in (train_set, test_set):
    set_.drop("wRC+_cat", axis=1, inplace=True)


# In[24]:


X_test = test_set.drop("WAR", axis=1) # drop labels for training set
y_test = test_set["WAR"].copy()


# In[25]:


X_test = X_test.drop("Name", axis=1)
X_test = X_test.drop("Team", axis=1)
X_test = X_test.drop("BB%", axis=1)
X_test = X_test.drop("K%", axis=1)


# In[26]:


corr_matrix = stat.corr()


# In[27]:


corr_matrix["WAR"].sort_values(ascending=False)


# In[28]:


attributes = ["WAR", "Off", "wRC+", "wOBA"]
pd.plotting.scatter_matrix(stat[attributes], figsize=(12,8))


# In[29]:


stat.plot(kind="scatter", x="wRC+", y="WAR", alpha=0.1)
plt.axis([0, 200, 0, 10])


# In[30]:


stat = train_set.drop("WAR", axis=1) 
stat_labels = train_set["WAR"].copy()


# In[31]:


stat_num = stat.drop("Name", axis=1)
stat_num = stat_num.drop("Team", axis=1)
stat_num = stat_num.drop("BB%", axis=1)
stat_num = stat_num.drop("K%", axis=1)


# In[32]:


num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])


# In[33]:


stat_num.head()


# In[34]:


imputer = SimpleImputer(strategy="median")


# In[35]:


imputer.fit(stat_num)


# In[36]:


imputer.strategy


# In[37]:


imputer.statistics_


# In[38]:


stat_num.median().values


# In[39]:


X = imputer.transform(stat_num)


# In[40]:


stat_tr = pd.DataFrame(X, columns = stat_num.columns, index=stat.index)


# In[44]:


stat_num_tr = num_pipeline.fit_transform(stat_num)
stat_num_tr


# In[45]:


stat_num_tr.shape


# In[46]:


lin_reg = LinearRegression()
lin_reg.fit(stat_num_tr, stat_labels)


# In[47]:


stat = stat.drop("Name", axis=1)
stat = stat.drop("Team", axis=1)
stat = stat.drop("BB%", axis=1)
stat = stat.drop("K%", axis=1)


# In[48]:


some_data = stat.iloc[:5]
some_data_prepared = num_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))


# In[49]:


stat_predictions = lin_reg.predict(stat_num_tr)
lin_mse = mean_squared_error(stat_labels, stat_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[50]:


lin_mae = mean_absolute_error(stat_labels, stat_predictions)
lin_mae


# In[51]:


from sklearn.tree import DecisionTreeRegressor


# In[52]:


tree_reg = DecisionTreeRegressor(random_state=50)
tree_reg.fit(stat_num_tr, stat_labels)


# In[53]:


some_data = stat.iloc[:5]
some_data_prepared = num_pipeline.transform(some_data)
print("Predictions:", tree_reg.predict(some_data_prepared))


# In[54]:


some_labels = stat_labels.iloc[:5]
print("Labels:", list(some_labels))


# In[55]:


stat_predictions = tree_reg.predict(stat_num_tr)
tree_reg_mse = mean_squared_error(stat_labels, stat_predictions)
tree_reg_rmse = np.sqrt(tree_reg_mse)
tree_reg_rmse


# In[56]:


tree_reg_mae = mean_absolute_error(stat_labels, stat_predictions)
tree_reg_mae


# In[57]:


lin_reg = LinearRegression()
lin_scores = cross_val_score(lin_reg, stat_num_tr, stat_labels,
                                scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-lin_scores)).describe()


# In[58]:


tree_reg = DecisionTreeRegressor(random_state=50)
tree_scores = cross_val_score(tree_reg, stat_num_tr, stat_labels,
                                scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-tree_scores)).describe()


# In[59]:


forest_reg = RandomForestRegressor(n_estimators=100, random_state=50)
forest_scores = cross_val_score(forest_reg, stat_num_tr, stat_labels,
                                scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-forest_scores)).describe()


# In[60]:


svm_reg = SVR(kernel="linear")
svm_scores = cross_val_score(svm_reg, stat_num_tr, stat_labels,
                                scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-svm_scores)).describe()


# In[61]:


param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]


# In[76]:


forest_reg = RandomForestRegressor(random_state=50)


# In[103]:


grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(stat_num_tr, stat_labels)


# In[104]:


grid_search.best_params_


# In[106]:


grid_search.best_estimator_


# In[107]:


pd.DataFrame(grid_search.cv_results_)


# In[108]:


feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances


# In[109]:


final_model = grid_search.best_estimator_


# In[110]:


num_pipeline_with_predictor = Pipeline([
        ("preparation", num_pipeline),
        ("final model", final_model)
    ])


# In[111]:


joblib.dump(num_pipeline_with_predictor, "my_model.pkl") # DIFF
my_model_loaded = joblib.load("my_model.pkl") # DIFF


# In[112]:


y_test_pred = my_model_loaded.predict(X_test)


# In[113]:


final_mse = mean_squared_error(y_test, y_test_pred)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[ ]:




