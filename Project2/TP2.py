# -*- coding: utf-8 -*-
"""

@author: Diogo Rodrigues 56153 && Jose Murta 55226
"""

import tp2_aux as aux
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
import numpy as np
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

##Functions
def pca (matrix):
    pca = PCA(n_components=6)
    pca.fit(matrix)
    t_data = pca.transform(matrix)
    return t_data

def kernel_pca (matrix):
    kernel_pca = KernelPCA(n_components=6, kernel = 'rbf')
    kernel_pca.fit(matrix)
    t_data = kernel_pca.transform(matrix)
    return t_data

def iso (matrix):
    iso = Isomap(n_components=6)
    iso.fit(matrix)
    t_data = iso.transform(matrix)
    return t_data

def stds (d):
    d = (d - np.mean(d)) / np.std(d)
    return d

def agg_clust (n_clusters, matrix):
    ward = AgglomerativeClustering(n_clusters=n_clusters)
    pred = ward.fit_predict(matrix)
    return pred

def spectral_clust(n_clusters, matrix):
    clustering = SpectralClustering(n_clusters=n_clusters,assign_labels='cluster_qr')
    pred = clustering.fit_predict(matrix)
    return pred

def k_means_clust(n_clusters, matrix):
    kmeans = KMeans(n_clusters=n_clusters)
    pred = kmeans.fit_predict(matrix)
    sse = kmeans.inertia_
    return pred, sse
    
def confusion_matrix(clusters, groups):
    tPositive = 0
    tNegative = 0
    fPositive = 0
    fNegative = 0
    counter = 0
    for i in range(len(clusters)):
        if(groups[i] != 0):
            for j in range(i+1,len(clusters)):
                if(groups[j] != 0):
                    counter+=1
                    if groups[i] == groups[j]:
                        if clusters[i] == clusters[j]:
                            tPositive+=1
                        else:
                            fNegative+=1
                    else:
                        if clusters[i] == clusters[j]:
                            fPositive+=1
                        else:
                            tNegative+=1
    return tPositive, tNegative, fPositive, fNegative, counter

def precision(tPositives, fPositives):
    return tPositives / (tPositives+fPositives)

def recall(tPositives, fNegatives):
    return tPositives / (tPositives+fNegatives)

def rand(tPositives, tNegatives, total):
    numerator =tPositives + tNegatives
    return numerator / total

def f1(precision, recall):
    return 2* ((precision*recall)/(precision+recall))

def purity(n_clusters, clusters, labelsLabeled):
    total = 0
    for i in range(n_clusters):
        nclass1 = 0
        nclass2 = 0
        nclass3 = 0
        for j in range(len(clusters)):
            if (clusters[j] == i):
                if (labels[j] == 1):
                    nclass1+=1
                elif (labels[j] == 2):
                    nclass2+=1
                elif (labels[j] == 3):
                    nclass3+=1
        total += max (nclass1,nclass2,nclass3)
    return total/labelsLabeled
        
def returnExternalIndexes(clust_pred, labels, n_clusters, labelsLabeled):
        tPositive, tNegative, fPositive, fNegative, counter = confusion_matrix(clust_pred, labels)
        precision_aux = precision(tPositive, fPositive)
        recall_aux = recall (tPositive,fNegative)
        rand_aux = rand (tPositive,tNegative,counter)
        f1_aux = f1 (precision_aux,recall_aux)
        purity_aux = purity(n_clusters, clust_pred, labelsLabeled)
        return precision_aux, recall_aux, rand_aux, f1_aux, purity_aux
    

##Main code
labels = np.loadtxt("labels.txt", delimiter= ",")
labels = labels [:,1]
image_matrix = aux.images_as_matrix()
labelsLabeled = len(labels[labels[:]>0])

PCA_features = pca(image_matrix)
kernelPCA_features = kernel_pca(image_matrix)
iso_features = iso(image_matrix)
image_features = np.hstack((PCA_features, np.hstack((kernelPCA_features,iso_features))))

image_features =stds(image_features)

#Code inspired in the SkLearn documentation
selector = SelectKBest(f_classif, k=5)
X_train = []
y_train = []
for i in range (len(labels)):
    if labels[i] != 0:
        X_train.append(image_features[i])
        y_train.append(labels[i])

selector.fit(X_train, y_train)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()

X_indices = np.arange(image_features.shape[-1])
plt.figure(1)
plt.clf()
plt.bar(X_indices, scores, width=0.2)
plt.title("Feature univariate score")
plt.xlabel("Feature number")
plt.ylabel("Univariate score ($-Log(p_{value})$)")
plt.xticks(np.arange(0,18,1))
plt.show()

#After analysis of the plot we concluded that there are 5 features with significant values then k=5 in the SelectKBest. 
image_features = selector.transform(image_features) 


df = pd.DataFrame(image_features, columns = np.arange(0,5,1))
pd.plotting.scatter_matrix(df, hist_kwds={'bins':30})
plt.suptitle('Scatter matrix')
plt.show()

#After visual analysis of the scatter matrix plot, we concluded that that features 0 and 3, from
#the 5 best selected are redundant
image_features = image_features[:, 1:5]


agg_matrix = np.zeros((5, 9))
spectral_matrix = np.zeros((5, 9))
kmeans_matrix = np.zeros((5, 9))
kmeansloss = []

###Clustering
clusterArray = np.arange(2, 11, 1)
for n in clusterArray:
    print("--------------------CLUSTERS =", n, "--------------------")
    #Agglomerative
    agg_clust_pred = agg_clust(n, image_features)
    precision_agg, recall_agg, rand_agg, f1_agg, purity_agg = returnExternalIndexes(agg_clust_pred, labels, n, labelsLabeled)
    print("-----Agglomoretive with", n, "clusters-------")
    print("Precision:", precision_agg)
    agg_matrix[0][n-2] = precision_agg
    print("Recall:", recall_agg)
    agg_matrix[1][n-2] = recall_agg
    print("Rand:", rand_agg)
    agg_matrix[2][n-2] = rand_agg
    print("F1:", f1_agg)
    agg_matrix[3][n-2] = f1_agg
    print("Purity:", purity_agg)
    agg_matrix[4][n-2] = purity_agg
    
    #Our chosen number of clusters to use the report clusters to show in the html
    if n == 6:
    	aux.report_clusters(np.array(range(image_features.shape[0])), agg_clust_pred, "testAgg6.html")
    
    #Spectral
    spectral_clust_pred = spectral_clust(n, image_features)
    precision_spe, recall_spe, rand_spe, f1_spe, purity_spe = returnExternalIndexes(spectral_clust_pred, labels, n, labelsLabeled)
    print("-----Spectral with", n, "clusters-------")
    print("Precision:", precision_spe)
    spectral_matrix[0][n-2] = precision_spe
    print("Recall:", recall_spe)
    spectral_matrix[1][n-2] = recall_spe
    print("Rand:", rand_spe)
    spectral_matrix[2][n-2] = rand_spe
    print("F1:", f1_spe)
    spectral_matrix[3][n-2] = f1_spe
    print("Purity:", purity_spe)
    spectral_matrix[4][n-2] = purity_spe
    
    #Our chosen number of clusters to use the report clusters to show in the html
    if n == 6:
        aux.report_clusters(np.array(range(image_features.shape[0])), spectral_clust_pred, "testSpe6.html")
    
    #K-means
    kmeans_clust_pred, sse_kmeans = k_means_clust(n, image_features)
    precision_kmeans, recall_kmeans, rand_kmeans, f1_kmeans, purity_kmeans = returnExternalIndexes(kmeans_clust_pred, labels, n, labelsLabeled)
    print("-----K-means with", n, "clusters-------")
    print("Precision:", precision_kmeans)
    kmeans_matrix[0][n-2] = precision_kmeans
    print("Recall:", recall_kmeans)
    kmeans_matrix[1][n-2] = recall_kmeans
    print("Rand:", rand_kmeans)
    kmeans_matrix[2][n-2] = rand_kmeans
    print("F1:", f1_kmeans)
    kmeans_matrix[3][n-2] = f1_kmeans
    print("Purity:", purity_kmeans)
    kmeans_matrix[4][n-2] = purity_kmeans
    print("K-means loss / SSE:", sse_kmeans)
    kmeansloss.append(sse_kmeans)
    
    #Our chosen number of clusters to use the report clusters to show in the html
    if n==6:
        aux.report_clusters(np.array(range(image_features.shape[0])), kmeans_clust_pred, "testKmeans6.html")
    

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
fig.subplots_adjust(right=1.5, hspace=0.4)
fig.suptitle('External Indexes')
titleArray = ["Precision", "Recall", "Rand", "F1", "Purity"]
counter = 0

for i in range(0,2):
    for j in range(0,3):
        axs[i,j].set_title(titleArray[counter])
        axs[i,j].set_xlabel('Number of clusters')
        axs[i,j].set_ylabel(titleArray[counter])
        agg_legend = mlines.Line2D([], [], color='red', marker='_', label='Agglomerative', linestyle ='None')
        spectral_legend = mlines.Line2D([], [], color='blue', marker='_', label='Spectral', linestyle ='None')
        kmeans_legend = mlines.Line2D([], [], color='green', marker='_', label='K-means', linestyle ='None')
        axs[i,j].legend(handles=[agg_legend, spectral_legend, kmeans_legend])
        
        axs[i,j].plot(clusterArray, agg_matrix[counter], '-r') 
        axs[i,j].plot(clusterArray, spectral_matrix[counter], '-b')
        axs[i,j].plot(clusterArray, kmeans_matrix[counter], '-g')
        
        
        counter+=1
        if counter == 5:
            break


# Plot the k means loss
fig, ax = plt.subplots()
plt.xlabel('Number of clusters')
plt.ylabel('K-means loss')
plt.plot(clusterArray, kmeansloss, '-b')
plt.show()


##DBSCAN with the data set to anwer question Q6
print ("-----------------DBSCAN----------QUESTION6")

#Inspirede in sklearn documentation
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors # importing the library
from sklearn import metrics

neighb = NearestNeighbors(n_neighbors=8) 
nbrs=neighb.fit(image_features)
distances,indices=nbrs.kneighbors(image_features)

distances = np.sort(distances, axis = 0) 
distances = distances[:, 1] 
plt.plot(distances)
plt.title ("Optimal eps using KNN method")
plt.show()


#From the plot we can observe that the point of maximum curvature is located around
#optimum epsilon reaches 0,8
#Min Samples we used 8 because it is image_Features dimensions * 2 as recommended by several papers
db = DBSCAN(eps=0.8, min_samples=8).fit(image_features)
labels = db.labels_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters of DBSCAN: %d" % n_clusters_)
print("Estimated number of noise points of DBSCANS: %d" % n_noise_)  
print(f"Silhouette Coefficient of DBSCAN: {metrics.silhouette_score(image_features, labels):.3f}")













