#!/usr/bin/env python
# coding: utf-8

# In[5]:


def fib(n):
    x=1
    y=1
    i=0
    print(x)
    print(y)
    while i<(n-2):
        
        z=x+y
        print(z)
        x=y
        y=z
        i=i+1


# In[6]:


fib(6)


# In[ ]:




