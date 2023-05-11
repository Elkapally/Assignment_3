#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import cluster_tools as ct
import errors as err
plt.rcParams['figure.figsize'] = [8,8]


# In[2]:


def read_process_data(file1, file2):
    df = pd.read_csv(file1)
    df1 = pd.read_csv(file2)
    # MELT
    df = df.melt(id_vars = 'date', var_name = 'country', value_name = 'mortality')
    df1 = df1.melt(id_vars = 'date', var_name = 'country', value_name = 'cereal')
    boolmask = (df1.cereal.notna()) & (df.mortality.notna())
    df = df.loc[boolmask, :].reset_index(drop = True)
    df1 = df1.loc[boolmask, :].reset_index(drop = True)
    df_main = pd.DataFrame({"mortality":df.mortality, 
                            'cereal':df1.cereal})
    df_scaled, df_min, df_max = ct.scaler(df_main)
    return df, df1, df_main, df_scaled, df_min, df_max

df, df1, df_main, df_scaled, df_min, df_max = read_process_data(file1 = 'Mortality rate, under-5 (per 1,000 live births).csv', 
                                                                file2 = 'Cereal yield (kg per hectare).csv')


# In[3]:


ct.map_corr(df_main)


# In[4]:


df_main.corr()


# In[59]:


def plot_inertia(df_scaled):
    clusters_inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters = k, random_state = 42)
        kmeans.fit(df_scaled)
        clusters_inertia.extend([kmeans.inertia_])
    plt.plot(range(1, 11), clusters_inertia, 'go-')
    plt.title('Elbow Plot')
    plt.xlabel('k'), plt.ylabel('inertia')
    plt.grid(alpha = 0.5)
    plt.show()
plot_inertia(df_scaled)


# In[11]:


def visualize_clusters(dfmain, dfsc, k):
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(dfsc)
    # back scale the cluster centres
    cluster_centers = ct.backscale(kmeans.cluster_centers_, df_min, df_max)
    cols = ['darkblue', 'darkgreen', 'red', 'purple'] # colors
    dfmain['kmeanslabel'] = kmeans.labels_
    for i in range(k):
        mask = dfmain['kmeanslabel'] == i
        y, x = dfmain.loc[mask, 'mortality'], dfmain.loc[mask, 'cereal']
        plt.scatter(x, y, color = cols[i])
        plt.scatter(cluster_centers[i, 1], cluster_centers[i, 0], 
                    marker = 'o', s = 150, color = 'yellow')
    plt.xlabel('Cereal Yield (kg per hectare)')
    plt.ylabel('Mortality (per 1,000 live births)')
    plt.title('Cereal Yield Production against Mortality')
    plt.grid(alpha = 0.5)
    plt.show()
visualize_clusters(df_main, df_scaled, k = 4)


# In[24]:


def visualize_each_cluster(df, dfmain, k, v = True):    
    mask = (dfmain.kmeanslabel == k)
    dates = df.date[mask]
    unique_countries = list(np.unique(df.country[mask]))
    dfmortality = dfmain.loc[mask, ['mortality', 'cereal']]
    dfmortality['date'] = dates
    dfmortality = dfmortality.groupby('date').agg({'mortality':'mean', 'cereal':'mean'})
    print(f"{len(unique_countries)} Countries Present in Cluster {k+1} Are\n-----------------------------\n{', '.join(unique_countries)}\n")
    if v == True: # Plot
        dfmortality.plot('cereal', 'mortality', kind = 'scatter')
        plt.ylabel('Mortality (per 1,000 live births)')
        plt.grid(alpha = 0.5)
        plt.title(f"Cluster {k+1}")
        plt.show()
    else:
        return dfmortality

for i in [1, 3]:
    visualize_each_cluster(df, df_main, k = i-1)


# In[57]:


# The first curve seems linear, the second one takes an Inverted C hyperbola
def linear_func_cluster1(x, a, b):
    return a*x + b

def fit_cluster1(df):
    xdata = df['cereal'].to_numpy()
    ydata = df['mortality'].to_numpy()
    idx = np.argsort(xdata) # Sort by indexes
    xdata = xdata[idx]
    ydata = ydata[idx]
    ests, covs = curve_fit(linear_func_cluster1, xdata, ydata)
    yest = linear_func_cluster1(xdata, *ests)
    sigma = np.sqrt(np.diag(covs)) 
    # Plot
    df.plot('cereal', 'mortality', kind = 'scatter', label = 'Actual Mortality Rates')
    plt.plot(xdata, yest, color = 'purple', label = "Fitted Mortality Rates")
    if all(sigma > 1): # plot error confidence is sigmas are greater than 1
        yl, yu = err.err_ranges(xdata, linear_func_cluster1, ests, sigma)
        plt.fill_between(xdata, yl, yu, alpha = 0.3, label = 'Confidence Range')
    plt.title("Estimation of Mortality Rates Using Cereal Data - Cluster 1")
    plt.ylabel('Mortality Rates')
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.show()
dfm1 = visualize_each_cluster(df, df_main, 0, v = False)
print("-----------------------------------------------\nYears\n-------------------------------------------------")
print(', '.join(map(lambda x: str(x), dfm1.index.tolist())))
fit_cluster1(dfm1)


# In[56]:


def poly_func_cluster3(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def fit_cluster3(df):
    xdata = df['cereal'].to_numpy()
    ydata = df['mortality'].to_numpy()
    idx = np.argsort(xdata) # Sort by indexes
    xdata = xdata[idx]
    ydata = ydata[idx]
    ests, covs = curve_fit(poly_func_cluster3, xdata, ydata)
    yest = poly_func_cluster3(xdata, *ests)
    sigma = np.sqrt(np.diag(covs)) 
    # Plot
    df.plot('cereal', 'mortality', kind = 'scatter', label = 'Actual Mortality Rates')
    plt.plot(xdata, yest, color = '#33ffd7', label = "Fitted Mortality Rates")
    if all(sigma > 1): # plot error confidence is sigmas are greater than 1
        yl, yu = err.err_ranges(xdata, poly_func_cluster3, ests, sigma)
        plt.fill_between(xdata, yl, yu, alpha = 0.3, label = 'Confidence Range')
    plt.title("Estimation of Mortality Rates Using Cereal Data - Cluster 3")
    plt.ylabel('Mortality Rates')
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.show()
dfm3 = visualize_each_cluster(df, df_main, 2, v = False)
print("-----------------------------------------------\nYears\n-------------------------------------------------")
print(', '.join(map(lambda x: str(x), dfm3.index.tolist())))
fit_cluster3(dfm3)


# In[ ]:




