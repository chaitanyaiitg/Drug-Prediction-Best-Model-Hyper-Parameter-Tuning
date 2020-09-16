#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[23]:


df=pd.read_csv('F:/Data_Science/Drug/Drug.csv')


# In[24]:


df


# In[25]:


df.shape


# In[26]:


df.columns


# In[27]:


df.dtypes


# In[28]:


df.isnull().sum()


# In[29]:


df.nunique()


# In[30]:


missing_percentage=((df.isnull().sum()*100)/df.shape[0])
print(missing_percentage)


# In[31]:


df.describe()


# In[32]:


df.loc[:,df.isnull().any()]


# In[33]:


sns.heatmap(df.isnull())


# In[34]:


df['Drug'].unique()


# In[35]:


df['Drug'].replace({'DrugY': 'drugY'},inplace=True)


# In[36]:


df.head(20)


# In[37]:


df.dtypes


# In[38]:


df['Drug'].value_counts().sort_index()


# In[39]:


plt.figure(figsize=(12,7))
colors=colors=['r','g','b','m','c','y','teal','thistle','tomato','turquoise','violet','yellowgreen','coral','cornflowerblue','crimson','cyan','darkblue','darkcyan','darkslategray','darkturquoise','khaki','b','lawngreen','m','lightcoral','y','darkgreen','darkkhaki','darkmagenta','darkolivegreen','gold','goldenrod','darkred','darksalmon','darkseagreen','darkslateblue','darkviolet','deeppink','deepskyblue','darkgray','dodgerblue','firebrick','skyblue','forestgreen','fuchsia','pink','green','gray','green','greenyellow','red','hotpink','indianred']

df['Drug'].value_counts().sort_index().plot(kind='bar', color=colors,title=('Drugs'))


# In[40]:


df.columns


# In[41]:


sns.countplot(x='Drug', data=df, hue='Sex')


# In[42]:


sns.countplot(x='Drug', data=df, hue='Cholesterol',palette='Set2')


# In[43]:


sns.countplot(x='Drug', data=df, hue='BP',palette='Set3')


# In[44]:


df.columns


# In[45]:


sns.lmplot(x='Age',y='Na_to_K', data=df)


# In[46]:


plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
sns.swarmplot(x='Drug',y='Na_to_K', data=df)

plt.subplot(1,4,2)
sns.swarmplot(x='Drug',y='Na_to_K', data=df,hue='Sex')


plt.subplot(1,4,3)
sns.swarmplot(x='Drug',y='Na_to_K', data=df,hue='BP')

plt.subplot(1,4,4)
sns.swarmplot(x='Drug',y='Na_to_K', data=df,hue='Cholesterol')


#point", "bar", "strip", "swarm","box", "violin", or "boxen".


# In[47]:


df.head(4)


# In[ ]:





# In[48]:


from sklearn.preprocessing import LabelEncoder

# creating an encoder
encoder= LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
#df['Drug'] = encoder.fit_transform(df['Drug'])


# In[49]:


df.head(50)


# In[50]:


df['Drug']=df['Drug'].astype('category')


# In[51]:


df.dtypes


# In[52]:


df['Drug'].unique()


# In[53]:


X=df.drop(['Drug'],axis=1)
X


# In[54]:


y=df['Drug']
y


# In[55]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr(),cmap='RdYlGn',annot=True)
plt.show()


# In[56]:


from sklearn.ensemble import ExtraTreesClassifier
selection = ExtraTreesClassifier()
selection.fit(X,y)


# In[57]:


print(selection.feature_importances_)


# In[58]:


plt.figure(figsize=(15,10))
feature_imp=pd.Series(selection.feature_importances_, index=X.columns)
feature_imp.nlargest(20).plot(kind='barh')


# In[132]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[133]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=10)


# In[134]:


s=StandardScaler()
X_train=s.fit_transform(X_train)
X_test=s.transform(X_test)


# In[135]:


svc=SVC()
svc.fit(X_train,y_train)
svc.score(X_test,y_test)


# In[136]:


value=[]
algo=[]
def cross_val_score_model(model,name):
    cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
    cv_score=cross_val_score(model,X,y,cv=cv).mean()
    print('CV_Score' + ' '+ str(model) +': '+ str(cv_score))
    value.append(cv_score)
    algo.append(name)

cross_val_score_model(SVC(),'SVC')
cross_val_score_model(KNeighborsClassifier(),'KNeighbors Classifier')
cross_val_score_model(LogisticRegression(solver='liblinear',multi_class='auto'),'Logistic Regression')
cross_val_score_model(RandomForestClassifier(),'Random Forest Classifier')
cross_val_score_model(XGBClassifier(),'XGB Classifier')


# In[137]:



sns.barplot(x=value,y=algo)
plt.show()


# In[138]:



model_params={
    'random_forest':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[100,200,500,1000,1200],
            'criterion' : ['gini', 'entropy'],
            'max_features' : ['auto', 'sqrt'],
            'max_depth' : [5,10,15,20,25,30,35,40,45,50]
        }
    },
        'XGBoost':{
        'model':XGBClassifier(),
        'params':{
            'max_depth': [2, 3, 5, 10, 15,20,30],
            'booster':['gbtree','gblinear'],
            'learning_rate':[0.05,0.1,0.15,0.20,0.4],
            'min_child_weight':[1,2,3,4]
        }
    }
}


# In[139]:


scores=[]
cv = ShuffleSplit(n_splits=5, test_size=0.25, random_state=10)
for model_name, mp in model_params.items():
    random_clf=RandomizedSearchCV(mp['model'],mp['params'],cv=cv, return_train_score=False)
    random_clf.fit(X,y)
    scores.append({
        'model':model_name,
        'best_score':random_clf.best_score_,
        'best param':random_clf.best_params_,
        'best estimator':random_clf.best_estimator_
    })

ds=pd.DataFrame(scores,columns=['model','best_score','best param','best estimator'])
ds


# In[140]:


def display_text_max_col_width(df, width):
    with pd.option_context('display.max_colwidth', width):
        print(df)

display_text_max_col_width(ds['best param'], 800)


# In[141]:


cross_val_score_model(RandomForestClassifier(n_estimators=100,criterion='gini',max_features= 'auto', max_depth= 15),'Random Forest Classifier_Best')
cross_val_score_model(XGBClassifier(min_child_weight=2, max_depth=5,learning_rate= 0.15, booster= 'gbtree'),'XGB Classifier_Best')


# In[142]:


sns.barplot(x=value,y=algo)
plt.show()


# In[143]:


Best1=pd.Series(algo)
Best2=pd.Series(value)
pd.DataFrame({'Model':Best1,'CVScore':Best2})


# In[ ]:





# # Best Model RandomForest Classifier with Hyper Parameter Tuning

# In[149]:


from sklearn.metrics import accuracy_score
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
rf_y_pred=rf.predict(X_test)
score=accuracy_score(y_test,rf_y_pred)
score


# In[151]:


rf_best=RandomForestClassifier(n_estimators=100,criterion='gini',max_features= 'auto', max_depth= 15)
rf_best.fit(X_train,y_train)
rf_best_y_pred=rf_best.predict(X_test)
score1=accuracy_score(y_test,rf_best_y_pred)
score1


# In[152]:


df.iloc[1]


# In[153]:


a=[[47,1,1,0,13.093]]
a=s.transform(a)
b=rf_best.predict(a)
b


# In[157]:


from sklearn.metrics import confusion_matrix
c_m=confusion_matrix(y_test,rf_best_y_pred)
c_m


# In[ ]:


columns=['drugA','drugB','drugC','drugX','drugY']


# In[170]:


sns.heatmap(c_m, annot=True,cmap='RdYlGn', xticklabels = columns , yticklabels = columns)
plt.show()


# In[ ]:




