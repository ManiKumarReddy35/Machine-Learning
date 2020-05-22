#!/usr/bin/env python
# coding: utf-8

# ## **IMPORTING ALL THE REQUIRED MODULES**

# In[310]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import seaborn as sb
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import scikitplot as skplt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import time


# ## **LOADING THE DATA SET**

# In[311]:


total_data = load_wine()


# ## **CONVERTING THE SCIKIT LEARN DATA SET INTO PANDAS DATAFRAME**

# In[312]:


data = pd.DataFrame(total_data.data, columns=total_data.feature_names)
#https://stackoverflow.com/questions/38105539/how-to-convert-a-scikit-learn-dataset-to-a-pandas-dataset
data_y = pd.DataFrame(total_data.target,columns=['target'])
print("x-data shape::",data.shape)
print("y-data shape::",data_y.shape)


# ## **PREPARING DATA FOR PAIR PLOTS**

# In[313]:


data['target'] = total_data.target
print("updated data-set shape::",data.shape)


# # **1. THE PAIR-PLOTS**

# In[ ]:


sb.pairplot(data,hue='target',kind="scatter",dropna=True)


# ## **COUNT OF DIFFERENT CLASSES PRESENT IN THE DATA SET**

# In[314]:


data['target'].value_counts().plot('bar')
plt.xlabel("CLASS LABEL")
plt.ylabel("COUNT IN THE DATA SET")
plt.show()


# ## **SPLITTING THE DATA INTO TRAIN AND TEST**
# 
# 
# 

# In[315]:


data_x = data.drop(['target'],axis=1)
print("data_x shape::",data_x.shape)


# In[316]:


x_train,x_test,y_train,y_test = train_test_split(data_x,data['target'],train_size=0.70,random_state=42,stratify=data['target'])
print("X train shape::",x_train.shape)
print("Y train shape::",y_train.shape)
print("X test shape::",x_test.shape)
print("Y test shape::",y_test.shape)


# ## **COUNT OF CLASSES IN THE Y-TRAIN**

# In[317]:


y_train.value_counts().plot('bar')
plt.xlabel("CLASS LABEL")
plt.ylabel("COUNT IN Y-TRAIN")
plt.show()


# ## **STANDARDIZING THE DATA**

# In[318]:


scale=StandardScaler(with_mean=True)
x_train_std = scale.fit_transform(x_train)
x_test_std = scale.transform(x_test)
print("x_train_std shape::",x_train_std.shape)
print("x_test_std shape::",x_test_std.shape)


# # **2. SVM**

# ## **SVM ONE V/S ONE**

# In[319]:


ovo_train = x_train.copy()
ovo_train = ovo_train.reset_index()
ovo_train.drop(['index'],axis=1,inplace=True)


# In[320]:


ovo_train.head()


# In[321]:


scaler = StandardScaler()


# In[322]:


ovo_train_std = pd.DataFrame(scaler.fit_transform(ovo_train),columns = ovo_train.columns)
ovo_train_std.head()


# In[323]:


yy = y_train.copy()
yy = yy.reset_index()
yy.drop(['index'],axis=1,inplace=True)


# In[324]:


ovo_train_std['target'] = yy.copy()


# In[325]:


ovo_train_0 = ovo_train_std[(ovo_train_std.target==0) | (ovo_train_std.target==1)]
ovo_train_1 = ovo_train_std[(ovo_train_std.target==1) | (ovo_train_std.target==2)]
ovo_train_2 = ovo_train_std[(ovo_train_std.target==2) | (ovo_train_std.target==0)]
len(ovo_train_0),len(ovo_train_1),len(ovo_train_2)


# In[326]:


ovo_train_0_x_std = ovo_train_0.drop(['target'],axis=1)
ovo_train_1_x_std = ovo_train_1.drop(['target'],axis=1)
ovo_train_2_x_std = ovo_train_2.drop(['target'],axis=1)


ovo_train_0_y = ovo_train_0['target']
ovo_train_1_y = ovo_train_1['target']
ovo_train_2_y = ovo_train_2['target']

len(ovo_train_0_x_std),len(ovo_train_0_y),len(ovo_train_1_x_std),len(ovo_train_1_y),len(ovo_train_2_x_std),len(ovo_train_2_y)


# In[327]:


svm_ovo_0 = SVC(C=1,kernel='linear',probability=True).fit(ovo_train_0_x_std,ovo_train_0_y)
svm_ovo_1 = SVC(C=1,kernel='linear',probability=True).fit(ovo_train_1_x_std,ovo_train_1_y)
svm_ovo_2 = SVC(C=1,kernel='linear',probability=True).fit(ovo_train_2_x_std,ovo_train_2_y)


# In[328]:


planeovo_0 = svm_ovo_0.coef_
planeovo_1 = svm_ovo_1.coef_
planeovo_2 = svm_ovo_2.coef_

inter_0 = svm_ovo_0.intercept_
inter_1 = svm_ovo_1.intercept_
inter_2 = svm_ovo_2.intercept_


# ## TEST PREDICTIONS

# In[329]:


from statistics import mode
preds_temp = []
preds_ovo = []
for i in range(len(x_test_std)):
    res0 = np.dot(x_test_std[i],planeovo_0.T)+inter_0
    if(res0<0):
        preds_temp.append(0)
    if(res0>0):
        preds_temp.append(1)
    res1 = np.dot(x_test_std[i],planeovo_1.T)+inter_1
    if(res1<0):
        preds_temp.append(1)
    if(res1>0):
        preds_temp.append(2)
    res2 = np.dot(x_test_std[i],planeovo_2.T)+inter_2
    if(res2<0):
        preds_temp.append(0)
    if(res2>0):
        preds_temp.append(2)
    #print(i,preds_temp)
    try:
        preds_ovo.append(mode(preds_temp))
    except:
        preds_ovo.append(l[1])
    preds_temp.clear()


# ## TRAINING PREDICTIONS FOR TRAINING ACCURACY

# In[330]:


ovo_train_std_tr = pd.DataFrame(scaler.fit_transform(ovo_train),columns = ovo_train.columns)


# In[331]:


preds_temp_tr = []
preds_ovo_tr = []
for i in range(len(ovo_train_std_tr)):
    res0 = np.dot(np.array(ovo_train_std_tr)[i],planeovo_0.T)+inter_0
    if(res0<0):
        preds_temp_tr.append(0)
    if(res0>0):
        preds_temp_tr.append(1)
    res1 = np.dot(np.array(ovo_train_std_tr)[i],planeovo_1.T)+inter_1
    if(res1<0):
        preds_temp_tr.append(1)
    if(res1>0):
        preds_temp_tr.append(2)
    res2 = np.dot(np.array(ovo_train_std_tr)[i],planeovo_2.T)+inter_2
    if(res2<0):
        preds_temp_tr.append(0)
    if(res2>0):
        preds_temp_tr.append(2)
    try:
        preds_ovo_tr.append(mode(preds_temp_tr))
    except:
        preds_ovo_tr.append(l[1])
    preds_temp_tr.clear()


# In[370]:


print("TRAINING ACCURACY USING SVM ONE V/S ONE::",metrics.accuracy_score(preds_ovo_tr,y_train)*100,"%")
print("TEST ACCURACY USING SVM ONE V/S ONE::",metrics.accuracy_score(preds_ovo,y_test)*100,"%")
class_wise_acc = confusion_matrix(y_test,preds_ovo).diagonal()/confusion_matrix(y_test,preds_ovo).sum(axis=1)
print("CLASS-0 ACCURACY::",class_wise_acc[0]*100,"%")
print("CLASS-1 ACCURACY::",class_wise_acc[1]*100,"%")
print("CLASS-2 ACCURACY::",class_wise_acc[2]*100,"%")
print("F-1 Score USING SVM ONE V/S ONE::",metrics.f1_score(y_test,preds_ovo,average='macro'))
svm_one_vs_one = SVC(C=1, kernel='linear',tol=0.01,max_iter=-1,decision_function_shape='ovo', random_state=None,probability=True)
svm_one_vs_one.fit(x_train_std,y_train)
test_prob_sovo = svm_one_vs_one.predict_proba(x_test_std)
skplt.metrics.plot_roc_curve(y_test,test_prob_sovo,title="ROC CURVE FOR 3 CLASSES USING SVM (ONE V/S ONE) CLASSIFIER",curves="each_class")
plt.show()


# In[1]:


pip install scikit-plot


# ## **SVM ONE V/S REST**

# In[344]:


x_train,x_test,y_train,y_test = train_test_split(data_x,data['target'],train_size=0.70,random_state=42,stratify=data['target'])
print("X train shape::",x_train.shape)
print("Y train shape::",y_train.shape)
print("X test shape::",x_test.shape)
print("Y test shape::",y_test.shape)


# In[345]:


scale=StandardScaler(with_mean=True)
x_train_std = scale.fit_transform(x_train)
x_test_std = scale.transform(x_test)
print("x_train_std shape::",x_train_std.shape)
print("x_test_std shape::",x_test_std.shape)


# ### FUNCTION DEFINITION FOR MODYFING LABELS ACCORDANCE WITH ONE VS REST

# In[346]:


def modlabel(y,tar):
    for i in range(len(y)):
        if(y.iloc[i]==tar):
            y.iloc[i]=1
        else:
            y.iloc[i]=0
    return y


# ### FUNCTION CALLING FOR CREATING MODIFIED LABELS FOR ONE VS REST

# In[347]:


y_train0 = modlabel(y_train.copy(),0)
y_train1 = modlabel(y_train.copy(),1)
y_train2 = modlabel(y_train.copy(),2)


# ### **CREATING THREE DIFFERENT LINEAR CLASSIFIERS TRAINED WITH MODIFIED LABELS**

# In[348]:


svm_0 = SVC(C=1,kernel='linear',probability=True).fit(x_train_std,y_train0)
svm_1 = SVC(C=1,kernel='linear',probability=True).fit(x_train_std,y_train1)
svm_2 = SVC(C=1,kernel='linear',probability=True).fit(x_train_std,y_train2)


# ### **RETRIEVING THE HYPER PLANES OF THREE CLASSIFIERS**

# In[349]:


plane0 = svm_0.coef_
plane1 = svm_1.coef_
plane2 = svm_2.coef_


# In[356]:


plane0.shape,x_test_std[0].shape


# # IMPLEMENTATION OF ONE V/S REST FOR SVM

# In[362]:


def onevsrest(plane0c,plane1c,plane2c,xdata,svm0inter,svm1inter,svm2inter):
    preds_one_rest=[]
    for i in range(len(xdata)):
        res0=np.dot(xdata[i],plane0.T)+svm0inter
        res1=np.dot(xdata[i],plane1.T)+svm1inter
        res2=np.dot(xdata[i],plane2.T)+svm2inter
        if(res0>res1 and res0>res2):
            preds_one_rest.append(0)
        if(res1>res0 and res1>res2):
            preds_one_rest.append(1)
        if(res2>res0 and res2>res1):
            preds_one_rest.append(2)
    return np.array(preds_one_rest)
#onevsrest(plane0,plane1,plane2,x_train_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_),y_train)


# ### **RETRIEVING THE PROBABILTIES WITH RESPECT TO EACH CLASS**

# In[363]:


prob0vsoth = svm_0.predict_proba(x_test_std)[:,1:]
prob1vsoth = svm_1.predict_proba(x_test_std)[:,1:]
prob2vsoth = svm_2.predict_proba(x_test_std)[:,1:]


# ### **CONCATENATING THE CLASS WISE PROBABILTIES COLUMN WISE INTO ONE ARRAY**

# In[364]:


probs=[]
for i in range(len(prob0vsoth)):
    probs.append(prob0vsoth[i])
    probs.append(prob1vsoth[i])
    probs.append(prob2vsoth[i])
probs_arr = np.array(probs).reshape(len(prob0vsoth),3)


# In[365]:


y_train.shape,x_train_std.shape


# In[366]:


print("TRAINING ACCURACY USING SVM ONE V/S REST::",metrics.accuracy_score(onevsrest(plane0,plane1,plane2,x_train_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_),np.array(y_train).reshape(124,1))*100,"%")
print("TEST ACCURACY USING SVM ONE V/S REST::",metrics.accuracy_score(onevsrest(plane0,plane1,plane2,x_test_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_),np.array(y_test).reshape(54,1))*100,"%")
class_wise_acc = confusion_matrix(y_test,onevsrest(plane0,plane1,plane2,x_test_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_)).diagonal()/confusion_matrix(y_test,onevsrest(plane0,plane1,plane2,x_test_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_)).sum(axis=1)
print("CLASS-0 ACCURACY::",class_wise_acc[0]*100,"%")
print("CLASS-1 ACCURACY::",class_wise_acc[1]*100,"%")
print("CLASS-2 ACCURACY::",class_wise_acc[2]*100,"%")
print("F-1 Score USING SVM ONE V/S REST::",metrics.f1_score(y_test,onevsrest(plane0,plane1,plane2,x_test_std.copy(),svm_0.intercept_,svm_1.intercept_,svm_2.intercept_),average='macro'))
skplt.metrics.plot_roc_curve(y_test,probs_arr,title="ROC CURVE FOR 3 CLASSES USING SVM (ONE V/S REST) CLASSIFIER",curves="each_class")
plt.show()


# # **3. GAUSSIAN NAIVE BAYES**

# In[372]:


gnb = GaussianNB()
start=time.time()
gnb.fit(x_train_std,y_train)
end=time.time()-start
print(time.strftime("TRAINING TIME::%H:%M:%S", time.gmtime(end)))
test_prob_nb = gnb.predict_proba(x_test_std)
prediction_nb=[]
prediction_nb = test_prob_nb.argmax(axis=1)
predictions_nb = np.array(prediction_nb)
print("TRAINING ACCURACY USING NAIVE BAYES::",gnb.score(x_train_std,y_train)*100,"%")
print("TEST ACCURACY USING NAIVE BAYES::",gnb.score(x_test_std,y_test)*100,"%")
class_wise_acc = confusion_matrix(y_test, predictions_nb).diagonal()/confusion_matrix(y_test, predictions_nb).sum(axis=1)
print("CLASS-0 ACCURACY::",class_wise_acc[0]*100,"%")
print("CLASS-1 ACCURACY::",class_wise_acc[1]*100,"%")
print("CLASS-2 ACCURACY::",class_wise_acc[2]*100,"%")
print("F-1 Score USING NAIVE BAYES::",metrics.f1_score(y_test,predictions_nb,average='macro'))
skplt.metrics.plot_roc_curve(y_test,test_prob_nb,title="ROC CURVE OF 3 CLASSES USING NAIVE BAYES CLASSIFIER",curves="each_class")
plt.show()


# 
# # **4. DECISION TREES**

# ## **HYPER PARAMETER TUNING OF DEPTH OF TREE USING CV DATA**

# ### **SPLTTING THE DATA INTO TRAIN,CV**

# In[ ]:


x_train_std,x_cv_std,y_train,y_cv = train_test_split(x_train_std,y_train,train_size=0.80,stratify=y_train)
print("x_train_std shape::",x_train_std.shape)
print("x_cv_std shape::",x_cv_std.shape)
print("y_train shape::",y_train.shape)
print("y_cv shape::",y_cv.shape)


# In[ ]:


from sklearn import metrics
depths = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
scores=[]
for i in depths:
    dec_tree = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,max_depth=i)
    dec_tree.fit(x_train_std,y_train)
    pred = dec_tree.predict(x_cv_std)
    scores.append(metrics.accuracy_score(y_cv,pred))
plt.plot(depths,scores)
plt.xlabel("DEPTH OF THE TREE")
plt.ylabel("SCORE ON CV DATA")
plt.show()


# In[ ]:


dec_tree = DecisionTreeClassifier(criterion="gini",min_samples_leaf=2,max_depth=9)
start = time.time()
dec_tree.fit(x_train_std,y_train)
end=time.time()-start
print(time.strftime("TRAINING TIME::%H:%M:%S", time.gmtime(end)))
test_prob_dtree = dec_tree.predict_proba(x_test_std)
prediction_dtree=[]
prediction_dtree = test_prob_dtree.argmax(axis=1)
predictions_dtree = np.array(prediction_dtree)
print("TRAINING ACCURACY USING DECISON TREE CLASSIFIER::",dec_tree.score(x_train_std,y_train)*100,"%")
print("TEST ACCURACY USING DECISON TREE CLASSIFIER::",dec_tree.score(x_test_std,y_test)*100,"%")
class_wise_acc = confusion_matrix(y_test, predictions_dtree).diagonal()/confusion_matrix(y_test, predictions_dtree).sum(axis=1)
print("CLASS-0 ACCURACY::",class_wise_acc[0]*100,"%")
print("CLASS-1 ACCURACY::",class_wise_acc[1]*100,"%")
print("CLASS-2 ACCURACY::",class_wise_acc[2]*100,"%")
print("F-1 Score USING DECISION TREE CLASSIFIER::",metrics.f1_score(y_test,predictions_dtree,average='macro'))
skplt.metrics.plot_roc_curve(y_test,test_prob_dtree,title="ROC CURVE FOR 3 CLASSES USING DECISION TREE CLASSIFIER",curves="each_class")
plt.show()


# In[ ]:




