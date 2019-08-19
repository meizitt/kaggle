#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

#处理数据
music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre'])
y = music_data['genre']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
)

#建立模型
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#保存模型
joblib.dump(model, 'music-recommender.joblib')

#输入测试数据，输出预测结果
predictions = model.predict(X_test)

#计算准确率
score = accuracy_score(y_test, predictions)
score


# In[48]:


#载入训练好的模型，输入要预测的数据[age,gender]
model = joblib.load('music-recommender.joblib')
predictions = model.predict([[21, 1]])
predictions


# In[50]:


from sklearn import tree

#以dot格式导出决策树
tree.export_graphviz(model,
                     out_file='music-recommender.dot',
                     feature_names=['age', 'gender'],
                     class_names=sorted(y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)

