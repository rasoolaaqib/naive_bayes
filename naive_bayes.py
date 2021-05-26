#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
from natsort import natsorted
import re
import string
from collections import Counter
import timeit
import matplotlib.pyplot as plt
import math
import collections
import operator
import time


# In[2]:


#reading train data (positive)
a = os.listdir('train/pos')
a = natsorted(a)
train_pos = []
for x in a:
    file_path = "train/pos/"+x
    f = open(file_path,'r+',encoding = "utf8")
    read = f.read()
    train_pos.append(read)
f.close()


# In[3]:


#reading train data (negative)
a = os.listdir('train/neg')
a = natsorted(a)
train_neg = []
for x in a:
    file_path = "train/neg/"+x
    f = open(file_path,'r+',encoding = "utf8")
    read = f.read()
    train_neg.append(read)
    f.close()


# In[4]:


#reading test data (positive)
test_dir = os.listdir('test/pos')
test_dir = natsorted(test_dir)
test_pos = []
for x_test in test_dir:
    file_path = "test/pos/"+x_test
    f_test = open(file_path,'r+',encoding = "utf8")
    read = f_test.read()
    test_pos.append(read)
    f_test.close()


# In[5]:


#reading test data (negaitive)
test_dir = os.listdir('test/neg')
test_dir = natsorted(test_dir)
test_neg = []
for x_test in test_dir:
    file_path = "test/neg/"+x_test
    f_test = open(file_path,'r+',encoding = "utf8")
    read = f_test.read()
    test_neg.append(read)
    f_test.close()


# In[6]:


#reading stop words
f = open('stop_words.txt', 'r+')
stop_words = f.read().splitlines()


# In[7]:


#converting to lowercase
def conv_lc(x):
    x = list(map(lambda x: x.lower(), x))
    return x


# In[8]:


#removing stop words
def remove_stopwords(x):
    stop = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)))
    for i,j in enumerate(x):    
        x[i] = stop.sub("", j)
    return x


# In[9]:


#removing punctuations
def remove_punc(x):
    table = str.maketrans(dict.fromkeys(string.punctuation))
    for i, j in enumerate(x):
        x[i]=j.translate(table)
    return x


# In[10]:


#removing numbers
def remove_numbers(x):
    for i,j in enumerate(x):
        x[i]=re.sub("\b\d+\b", "", j)
    return x


# In[11]:


train_pos=conv_lc(train_pos)
train_neg=conv_lc(train_neg)

train_pos=remove_stopwords(train_pos)
train_neg=remove_stopwords(train_neg)

train_pos=remove_punc(train_pos)
train_neg=remove_punc(train_neg)

train_pos=remove_numbers(train_pos)
train_neg=remove_numbers(train_neg)


# In[12]:


test_pos=conv_lc(test_pos)
test_neg=conv_lc(test_neg)

test_pos=remove_stopwords(test_pos)
test_neg=remove_stopwords(test_neg)

test_pos=remove_punc(test_pos)
test_neg=remove_punc(test_neg)

test_pos=remove_numbers(test_pos)
test_neg=remove_numbers(test_neg)


# In[13]:


#building a DataFrame for train data
df_pos = pd.DataFrame(columns=['review'])
df_pos['review']=train_pos
df_pos['sentiment']="positive"

df_neg = pd.DataFrame(columns=['review'])
df_neg['review']=train_neg
df_neg['sentiment']="negative"

#train data DataFrame
df=pd.DataFrame()
df=df_pos
df = df.append(df_neg)
df


# In[14]:


#building a DataFrame for test data
df_pos_test = pd.DataFrame(columns=['review'])
df_pos_test['review']=test_pos
df_pos_test['sentiment']='positive'

df_neg_test = pd.DataFrame(columns=['review'])
df_neg_test['review']=test_neg
df_neg_test['sentiment']='negative'

#test data DataFrame
df_test=pd.DataFrame()
df_test=df_pos_test
df_test = df_test.append(df_neg_test)
df_test


# In[15]:


def train_naive_bayes(D,C):
    logprior = []
    V= []
    bigdoc=[]
    likelihood=[]
    count_pos=0
    count_neg=0
    #building vocabulary
    for x in D['review'].values:
        a = x.split()
        V.extend(a);
    V = list(set(V))
    #for each class
    for c in C:
        d=[]
        x=c
        N_doc = N_doc = len(D['review'])                          #number of documents in D
        N_c = len(D[D['sentiment']==c])                           #number of documents in D in class c
        logprior.append(np.log(N_c/N_doc))                        #calculating logprior
        for y in D[D['sentiment']==c]['review'].values:           #building bigdoc for both classes
            z = y.split()
            d.extend(z)
        d = list(d)
        bigdoc.append(d)                                          
    for i in range(2):                                            #calculating loglikelihood
        s = len(bigdoc[i])+len(V)
        e = {}
        count = Counter(bigdoc[i])
        for w in V:
            e[w] = math.log((count[w]+1)/s)
        likelihood.append(e)         
    return logprior,likelihood, V                                 #returning logprior, loglikelihood, and vocabulary


# In[16]:


classes = df['sentiment'].unique()
logprior,likelihood, V = train_naive_bayes(df,classes)           #training naive bayes


# In[17]:


def test_naive_bayes (testdoc,logprior,loglikelihood,C,V):
    sum["positive"]=logprior[0]
    sum["negative"]=logprior[1]
    z = testdoc.split()
    z = set(z)
    for y in z:
        try:
            sum["positive"]+=loglikelihood[0][y]
            sum["negative"]+=loglikelihood[1][y]
        except:
            continue                                            #if word not found in likelihood an exception will occur, if so move to next word in testdoc
    return max(sum.items(), key=operator.itemgetter(1))[0]      #returning key of the max value. i.e. 'positive' or 'negative'
        


# In[18]:


predicted=[]
sum = {"positive":0,"negative":0}
for x in list(df_test['review'].values):
    a = test_naive_bayes(x,logprior,likelihood,classes,V)      #testing our trained naive bayes
    predicted.append(a)                                        #giving us a list of predicted labels


# In[19]:


#calculating Accuracy
count = 0
for i in range(df_test['review'].values.size):
    if predicted[i] == df_test['sentiment'].values[i]:
        count+=1
print("My model Accuracy =",(count/df_test['sentiment'].values.size)*100,"%")


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[21]:


#using datasets from part 1
vectorizer = CountVectorizer()
train_p2 = vectorizer.fit_transform(df['review'].values)
test_p2=vectorizer.transform(df_test['review'].values)
NB=MultinomialNB()
NB.fit(train_p2,df['sentiment'].values)
p = NB.predict(test_p2)                                         #predicted labels


# In[22]:


#calculating Accuracy
accu = accuracy_score(df_test['sentiment'].values, p)
print("Sci-kit learn Accuracy =",accu*100,"%")


# In[23]:


#calculating the confusion matrix
cm = confusion_matrix(df_test['sentiment'].values, p)
print("Confusion matrix:")
print(cm)

