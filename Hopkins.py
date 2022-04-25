#importing the required libraries
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
from numpy.random import uniform


# function to compute hopkins's statistic for the dataframe X
def hopkins_statistic(X):
    
    X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures
    
    
    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    
    
    
    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
   
    
    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    
    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    
    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]
    
 
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    
    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H


#CALIBRAZIONE HOPKINS

n = 100000

v_h = np.zeros(n)

for i in range(n):
    s = np.random.uniform(size=1000)
    df = pd.DataFrame(s)
    v_h[i]=hopkins_statistic(df)

min_b = np.min(v_h)
max_b = np.max(v_h)


fig1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
ax1.hist(v_h)
ax1.set_xlabel('H',fontsize=20)
ax1.set_title("Distribuzione di H su un dataset uniforme generato 100000 volte",fontsize=30)
ax1.tick_params(labelsize=20)

v_h = np.sort(v_h)

m = int(n*0.95)

v_hm = np.zeros(m)

v_hm = v_h[int(n*0.025+1):int(n*0.975)]



"""

#GENERO GAUSSIANE PER CLUSTER

sigma1 = 0.01
sigma2 = 0.05
sigma3 = 0.1

x1 = 0.2
x2 = 0.7

stot=np.zeros(1000)
v_H = np.zeros(1000)

for k in range(1000):
    s1 = np.random.normal(x1,sigma2,500)
    s2= np.random.normal(x2,sigma2,500)
    for i in range(500):
        stot[i]=s1[i]
        stot[i+500]= s2[i]
    df = pd.DataFrame(stot)
    v_H[k]= hopkins_statistic(df)
    
fig1, ax1 = plt.subplots(1, 1, figsize=(15, 10))
ax1.hist(stot)
ax1.set_xlabel('Evento',fontsize=20)
ax1.set_title("Distribuzione di eventi cluster in un intervallo tra 0 e 1",fontsize=30)
ax1.tick_params(labelsize=20)


t=0

for i in range(1000):
    if v_H[i]<0.6:
        t=t+1


print(t/10)

"""

"""
#REPULSIONE



sigma1 = 0.001


prec = 100

stot=np.zeros(prec)
v_H = np.zeros(1000)
inter_std = np.zeros(1000)


for k in range(1000):

    for i in range(prec):
        stot[i]=np.random.normal(i/prec,sigma1,1)
    
    stot_corretto = stot[1:prec]
    inter =np.zeros(prec-2)
    for l in range(prec-2):
        inter[l]=stot_corretto[l+1]-stot_corretto[l]
    
    
    inter_std[k]= np.std(inter)/np.mean(inter)
    
    df = pd.DataFrame(stot_corretto)

    v_H[k]=hopkins_statistic(df)

fig2, ax2 = plt.subplots(1, 1, figsize=(15, 10))
ax2.hist(v_H)


fig3, ax3 = plt.subplots(1, 1, figsize=(15, 10))
ax3.hist(inter_std)



"""












    