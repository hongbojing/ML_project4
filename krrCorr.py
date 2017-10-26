
# coding: utf-8

# In[4]:


import numpy as np
from numpy import*
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from pylab import scatter, show, legend, xlabel, ylabel  
from sklearn.metrics import r2_score
import seaborn as sns

cm = sns.light_palette("green", as_cmap=True)

dataset = pd.read_csv("data_akbilgic.csv", header=0)
attributeList = ["ISE", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
yAttribute = "SP"
X = dataset[attributeList[:]]
X = np.array(X)
Y = dataset[[yAttribute]]
Y = np.array(Y)

# dax = dataset[attributeList["DAX"]]
# dax = np.array(dax)

# daxA = "DAX"
dax = dataset[["DAX"]]
dax = np.array(dax)

# print(dax)
# print(dataset["DAX"])


# In[59]:


dataset["DAX"].corr(dataset["EU"])


# In[60]:


dataset.drop(['SP'], axis=1).corr(method='spearman')


# In[61]:


dataset.drop(['SP'], axis=1).corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[53]:


dataset.drop(['SP'], axis=1).corr(method='spearman').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[54]:


dataset.drop(['SP'], axis=1).corr(method='kendall').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[55]:


dataset.corr(method='pearson').style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)


# In[70]:


dataset.corr(method='pearson').style.set_caption('Colormaps, with a caption.').background_gradient(cmap=cm)


# In[71]:


dataset.corr(method='kendall').style.set_caption('Colormaps, with a caption.').background_gradient(cmap=cm)


# In[73]:


dataset.corr(method='spearman').style.set_caption('Colormaps, with a caption.').background_gradient(cmap=cm)


# In[16]:


# plot corelated values
plt.rcParams['figure.figsize'] = [32, 4]

fig, ax = plt.subplots(nrows=1, ncols=8)

ax=ax.flatten()

cols = ['DAX', 'FTSE', 'BOVESPA', 'BOVESPA', 'EU', 'EM', 'ISE', 'NIKKEI']
colors = ['#415952', '#415952', '#415952', '#415952', '#415952', '#415952', '#415952', '#415952']
j = 0

for i in ax:
    if j==0:
        i.set_ylabel('SP')
    i.scatter(dataset[cols[j]], dataset['SP'], alpha=0.5, color=colors[j])
    i.set_xlabel(cols[j])
    i.set_title('Pearson: %s'%dataset.corr().loc[cols[j]]['SP'].round(2)+' Spearman: %s'%dataset.corr(method='spearman').loc[cols[j]]['SP'].round(2))
    j+=1
    
plt.show()


# In[77]:


# plot corelated values
plt.rcParams['figure.figsize'] = [16, 6]

fig, ax = plt.subplots(nrows=1, ncols=3)

ax=ax.flatten()

cols = ['BOVESPA', 'EU', 'EM']
colors = ['#415952', '#f35134', '#243AB5', '#243AB5']
j = 0

for i in ax:
    if j==0:
        i.set_ylabel('SP')
    i.scatter(dataset[cols[j]], dataset['SP'], alpha=0.5, color=colors[j])
    i.set_xlabel(cols[j])
    i.set_title('Pearson: %s'%dataset.corr().loc[cols[j]]['SP'].round(2)+' Spearman: %s'%dataset.corr(method='spearman').loc[cols[j]]['SP'].round(2))
    j+=1
    
plt.show()

