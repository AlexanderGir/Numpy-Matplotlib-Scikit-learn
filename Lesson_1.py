#!/usr/bin/env python
# coding: utf-8

# Тема “Вычисления с помощью Numpy”

# In[1]:


import numpy as np


# In[7]:


a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])
print(a)


# In[9]:


mean_a = np.mean(a, axis = 0)
print(mean_a)


# In[11]:


a_centered = np.array(a - mean_a)
print(a_centered)


# In[13]:


a_1=a_centered[:,0]
a_2=a_centered[:,1]
a_centered_sp = np.dot(a_1, a_2)
print(a_centered_sp)


# In[24]:


a.shape[0]


# In[22]:


a_centered_sp / (a.shape[0]-1)


# In[26]:


a_t = a.T
print(a_t)


# In[27]:


b = np.cov(a_t)
print(b)


# Тема “Работа с данными в Pandas”

# Задание 1

# In[28]:


import pandas as pd


# In[29]:


a = {
    "author_id": [1, 2, 3],
    "author_name": ['Тургенев', 'Чехов', 'Островский']
}

authors = pd.DataFrame(a)

authors


# In[31]:


b = {
    "author_id": [1, 1, 1, 2, 2, 3, 3],
    "book_title": ['Отцы и дети', 'Рудин', 'Дворянское гнездо', 'Толстый и тонкий', 'Дама с собачкой', 'Гроза', 'Таланты и поклонники'],
    "price": [450, 300, 350, 500, 450, 370, 290]
}

book = pd.DataFrame(b)

book


# Задание 2

# In[35]:


authors_price = pd.merge(authors, book, on='author_id', how='inner')

authors_price


# Задание 3

# In[38]:


top5 = authors_price.nlargest(5, 'price')

top5


# Задание 4

# In[54]:


df1 = authors_price.groupby('author_name').agg({'price': 'min'}).rename(columns={'price':'min_price'})
df2 = authors_price.groupby('author_name').agg({'price': 'max'}).rename(columns={'price':'max_price'})
df3 = authors_price.groupby('author_name').agg({'price': 'mean'}).rename(columns={'price':'mean_price'})

authors_stat = pd.concat([df1, df2, df3], axis = 1)

authors_stat


# Задание 5

# In[76]:


d = {
    "cover": ['твердая', 'мягкая', 'мягкая', 'твердая', 'твердая', 'мягкая', 'мягкая']
}
d = pd.DataFrame(d)
d


# In[77]:


authors_price = pd.concat([authors_price, d], axis = 1)

authors_price


# In[78]:


get_ipython().run_line_magic('pinfo', 'pd.pivot_table')


# In[79]:


book_info = pd.pivot_table(authors_price, values = 'price', index = ['author_name'],
                    columns = ['cover'], aggfunc = np.sum, fill_value = 0)
book_info


# In[80]:


book_info.to_pickle('book_info.pkl')


# In[81]:


book_info2 = pd.read_pickle('book_info.pkl')


# In[83]:


book_info


# In[84]:


book_info2

