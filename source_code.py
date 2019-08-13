#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas


# In[14]:


train_data_frame = pandas.read_csv("mnist_train.csv")
test_data_frame = pandas.read_csv("mnist_test.csv")


# In[15]:


train_data_frame.shape


# In[16]:


test_data_frame.shape


# In[17]:


train_data = train_data_frame.values


# In[19]:


X_train = train_data[ : ,1: ]


# In[20]:


Y_train = train_data[ : , 0]


# In[34]:


test_data = test_data_frame.values
X_test = test_data[:,1:]
Y_test = test_data[:,0]


# In[35]:


import numpy


# In[36]:


def cart_distance (x1, x2):
    return numpy.sqrt(sum((x1-x2)**2))


# In[37]:


def kNN (X, Y, test_point, k):
    vals = []
    m = X.shape[0]
    
    for i in range (m):
        dist = cart_distance(test_point, X[i])
        vals.append((dist, Y[i]))
        
    vals = sorted(vals)
    vals = vals[:k]
    
    vals = numpy.array(vals)
    
    new_vals = numpy.unique(vals[:,1], return_counts=True)
    
    index = new_vals[1].argmax()
    prediction = new_vals[0][index]
    
    return prediction


# In[38]:


result = kNN (X_train, Y_train, X_test[1], 5)


# In[39]:


print (int(result))


# In[49]:


print(Y_test[1])


# In[ ]:





# In[ ]:





# In[ ]:




