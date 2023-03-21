import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
data=pd.read_csv('heart_disease_data.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
num=data.select_dtypes(include='number')
print(num.columns.values)
q1=data['Data_Value'].quantile(0.25)
q3=data['Data_Value'].quantile(0.75)
iqr=q3-q1
u=q3+1.5*iqr
l=q1-1.5*iqr
df=data[(data['Data_Value']<u)&(data['Data_Value']>l)]

qu1=df['Data_Value'].quantile(0.25)
qu3=df['Data_Value'].quantile(0.75)
iqr_=qu3-qu1
up=qu3+1.5*iqr_
lo=qu1-1.5*iqr_
df=df[(df['Data_Value']<up)&(df['Data_Value']>lo)]

qua1=df['Data_Value'].quantile(0.25)
qua3=df['Data_Value'].quantile(0.75)
iq_r=qua3-qua1
upp=qua3+1.5*iq_r
low=qua1-1.5*iq_r
df=df[(df['Data_Value']<upp)&(df['Data_Value']>low)]

quan1=df['LowConfidenceLimit'].quantile(0.25)
quan3=df['LowConfidenceLimit'].quantile(0.75)
Iqr=quan3-quan1
uppe=quan3+1.5*Iqr
lowe=quan1-1.5*Iqr
df=df[(df['LowConfidenceLimit']<uppe)&(df['LowConfidenceLimit']>lowe)]

quant1=df['HighConfidenceLimit'].quantile(0.25)
quant3=df['HighConfidenceLimit'].quantile(0.75)
I_qr=quant3-quant1
upper=quant3+1.5*I_qr
lower=quant1-1.5*I_qr
df=df[(df['HighConfidenceLimit']<upper)&(df['HighConfidenceLimit']>lower)]

cat=data.select_dtypes(include='object')
print(cat.columns.values)
for i in cat.columns.values:
    if len(data[i].value_counts()) <=5:
        sn.countplot(data[i])
        plt.show()

for i in num.columns.values:
    for j in num.columns.values:
        plt.plot(df[i],marker='o',color='red',label=f'{i}')
        plt.plot(df[j],marker='x',color='blue',label=f"{j}")
        plt.title(f'{i} vs {j}')
        plt.legend()
        plt.show()

'''data[['Year','Data_Value','Data_Value_Alt',
      'LowConfidenceLimit',
 'HighConfidenceLimit' ,'LocationID']]=data[['Year','Data_Value',
                                             'Data_Value_Alt','LowConfidenceLimit',
 'HighConfidenceLimit' ,'LocationID']].fillna(data[['Year','Data_Value',
                                                    'Data_Value_Alt','LowConfidenceLimit',
 'HighConfidenceLimit' ,'LocationID']].mean())
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
df['topic']=lab.fit_transform(df['Topic'])
df['Break_Out_Category']=lab.fit_transform(df['Break_Out_Category'])
df['Break_Out_CategoryId']=lab.fit_transform(df['BreakOutCategoryId'])
df['breakOutId']=lab.fit_transform(df['BreakOutId'])
df['topicId']=lab.fit_transform(df['TopicId'])
df['data_Value_Unit']=lab.fit_transform(df['Data_Value_Unit'])
df['priorityArea1']=lab.fit_transform(df['PriorityArea1'])
df['priorityArea3']=lab.fit_transform(df['PriorityArea3'])

x=df[['Year','Data_Value','Data_Value_Alt','LowConfidenceLimit',
 'HighConfidenceLimit' ,'Break_Out_Category','priorityArea1','priorityArea3','data_Value_Unit','Break_Out_CategoryId','breakOutId','LocationID']]
y=df['topic']

plt.figure(figsize=(17,6))
corr = df.corr(method='kendall')
my_m=np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)

lr=LogisticRegression()
lr.fit(x_train,y_train)
print('Logistic regression',lr.score(x_test,y_test))

tre=DecisionTreeClassifier()
tre.fit(x_train,y_train)
print('Decision tree',tre.score(x_test,y_test))

xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print('XGB',xgb.score(x_test,y_test))


rf=RandomForestClassifier()
rf.fit(x_train,y_train)
print('Random forest',rf.score(x_test,y_test))

lgb=LGBMClassifier()
lgb.fit(x_train,y_train)'''
#print('Light GBM',lgb.score(x_test,y_test))'''