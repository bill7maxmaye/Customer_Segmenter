#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# In[117]:


data = pd.read_csv("D:/ML/CC GENERAL.csv")


# In[118]:


data


# In[119]:


data.dropna(inplace = True)


# In[120]:


data


# In[121]:


data.describe()


# In[136]:


column_headers = list(data.columns)
column_headers


# In[122]:


scaler = StandardScaler()


# In[145]:


#data[['BALANCE', 'BALANCE_FREQUENCY',	'PURCHASES',	'ONEOFF_PURCHASES',	'INSTALLMENTS_PURCHASES',	'CASH_ADVANCE',	'PURCHASES_FREQUENCY',	'ONEOFF_PURCHASES_FREQUENCY',	'PURCHASES_INSTALLMENTS_FREQUENCY',	'CASH_ADVANCE_FREQUENCY',	'CASH_ADVANCE_TRX',	'PURCHASES_TRX',	'CREDIT_LIMIT',	'PAYMENTS',	'MINIMUM_PAYMENTS',	'PRC_FULL_PAYMENT',	'TENURE']] = scaler.fit_transform(data[['BALANCE', 'BALANCE_FREQUENCY',	'PURCHASES',	'ONEOFF_PURCHASES',	'INSTALLMENTS_PURCHASES',	'CASH_ADVANCE',	'PURCHASES_FREQUENCY',	'ONEOFF_PURCHASES_FREQUENCY',	'PURCHASES_INSTALLMENTS_FREQUENCY',	'CASH_ADVANCE_FREQUENCY',	'CASH_ADVANCE_TRX',	'PURCHASES_TRX',	'CREDIT_LIMIT',	'PAYMENTS',	'MINIMUM_PAYMENTS',	'PRC_FULL_PAYMENT',	'TENURE']])
norm_data = normalize(data)
scaled_data = scaler.fit_transform(norm_data)

columns = []
for column_header in column_headers:
    columns.append(column_header)
#scaled_data = pd.DataFrame(scaled_data, columns=['BALANCE', 'BALANCE_FREQUENCY',	'PURCHASES',	'ONEOFF_PURCHASES',	'INSTALLMENTS_PURCHASES',	'CASH_ADVANCE',	'PURCHASES_FREQUENCY',	'ONEOFF_PURCHASES_FREQUENCY',	'PURCHASES_INSTALLMENTS_FREQUENCY',	'CASH_ADVANCE_FREQUENCY',	'CASH_ADVANCE_TRX',	'PURCHASES_TRX',	'CREDIT_LIMIT',	'PAYMENTS',	'MINIMUM_PAYMENTS',	'PRC_FULL_PAYMENT',	'TENURE'])
scaled_data = pd.DataFrame(scaled_data, columns = columns)


# In[146]:


scaled_data


# In[147]:


def identify_cluster_num(scaled_data, max_k):
    means = []
    inertias = []

    for k in range (1, max_k):
        kmeans = KMeans(n_clusters = k)
        kmeans.fit(scaled_data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    fig = plt.subplots(figsize = (10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()


# In[148]:


identify_cluster_num(scaled_data, 30)


# In[175]:


def optimized_k(opt_k):
    k_means = KMeans(
        n_clusters = opt_k,
        n_init = 'auto',
        random_state = 42
    )
    
    # Fit / Train the model
    k_means.fit(scaled_data)
    
    scaled_data['Cluster_num'] = k_means.labels_

optimized_k(10)


# In[176]:


scaled_data


# In[177]:


#sample two attributes cluster plot
def two_att_plot (att1, att2):
    plt.scatter(x = scaled_data[att1], y = scaled_data[att2], c = scaled_data['Cluster_num'])
    plt.xlim(-0.1, 1)
    plt.ylim(3, 1.5)
    plt.show()


# In[178]:


two_att_plot('BALANCE', 'PURCHASES')


# In[172]:


cluster_labels = k_means.labels_

# Apply t-SNE to reduce the dimensionality to 3
tsne = TSNE(
    n_components=3,
    random_state=42
)
tsne_result = tsne.fit_transform(scaled_data)
# Apply t-SNE to reduce the dimensionality to 2
tsne = TSNE(
    n_components=2,
    random_state=42
)
tsne_result_2d = tsne.fit_transform(scaled_data)
# Visualizing t-SNE reduction
tsne_3d_fig = px.scatter_3d(
    x = tsne_result[:, 0],
    y = tsne_result[:, 1],
    z = tsne_result[:, 2],
    color = cluster_labels,
    title = "t-SNE Cluster Visualization"
)
tsne_3d_fig.show()


# In[173]:


# Visualizing t-SNE reduction
tsne_2d_fig = px.scatter(
    x = tsne_result_2d[:, 0],
    y = tsne_result_2d[:, 1],
    color = cluster_labels,
    title = "t-SNE Cluster Visualization (2D)"
)
tsne_2d_fig.show()


# In[ ]:




