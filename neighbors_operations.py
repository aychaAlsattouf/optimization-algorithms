#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd 

# ## Neighbors Operations 

# ### Sawipping function

# In[5]:


def swapping_fun(elements ,N):
    
    swap1=random.randint(0,N-1)
    swap2=random.randint(0,N-1)
    while swap1 == swap2:
        swap2=random.randint(0,N-1)
    temp_elements=elements.copy()
    s1=elements[swap1]
    s2=elements[swap2]
    temp_elements[swap1],temp_elements[swap2]=s2,s1
    return temp_elements


# ### 2-opt function

# In[6]:


def opt_2_fun(elements ,N):
    #select index to change
    num1=random.randint(0,N-1)
    num2=random.randint(0,N-1)
    while num2 == num1:
        num2=random.randint(0,N-1)
    if num1 > num2 :
        temp = num1
        num1 = num2
        num2 = temp
   # print(elements[num1],"   ",elements[num2])
    temp_elements=elements.copy()
    temp_elements[num1:num2+1]=list(reversed(temp_elements[num1:num2+1]))
    return temp_elements


# ### Insertion function

# In[7]:


def insertion_fun(elements,N):
    num1=random.randint(0,N-1)
    num2=random.randint(0,N-1)
    while num2 == num1  :
        num2=random.randint(0,N-1)
    if num1 > num2 :
        temp = num1
        num1 = num2
        num2 = temp
    #print(elements[num1],"     ",elements[num2])
    temp_elements = elements.copy()
    a = elements[num1+1:num2].copy()
    temp_elements[num1:num2-1]=a
    temp_elements[num2] = elements[num1]
    temp_elements[num2-1] = elements[num2]
    return temp_elements

