import scipy as sp
import matplotlib.pyplot as plt

from kneed import KneeLocator
import math

import time

# For optimal transport operations:
import ot

# For pre-processing, normalization
from sklearn.preprocessing import StandardScaler, normalize

import numpy as np
import random, math, os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn, optim
from torch.autograd import grad
import torch.nn.functional as F




# For computing graph distances:
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, kneighbors_graph
from sklearn.metrics import DistanceMetric

def model(X,y,  epsilon =1e-3, tol= 1e-2, lr = 10, best_k = 5, n_neighbors = 10):
    
    print("model start")
    time1 = time.time()

    k_range = range(1, best_k)

    def kmeans_finder(X, k_range = k_range):
        """
        Find the best k value for K-Means clustering using the Elbow Method.

        Parameters:
        - X: Input data for clustering.
        - k_range: A range of k values to consider.

        Returns:
        - best_k: The best k value.
        """

        inertia = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            inertia.append(kmeans.inertia_)

        # Find the "elbow" point in the inertia plot
        diff = np.diff(inertia)
        diff_ratio = diff[:-1] / diff[1:]
        best_k = k_range[np.argmax(diff_ratio) + 1]

        
    
            
    
        # plot the WSS against the number of clusters with the elbow point

        # plt.plot(range(1, 5), wss)
        # plt.xlabel('Number of clusters')
        # plt.ylabel('Within-cluster sum of squares')
        # plt.title('Elbow method for optimal number of clusters')
        # plt.vlines(best_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        # plt.show()

        # print('Best number of clusters:', best_k)

        
        
        # fit the KMeans model with k clusters
        kmeans = KMeans(n_clusters= best_k, random_state=42).fit(X)
    
        # get the centroids and labels
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
    
        # loop through each centroid and find the index of the closest point
        closest_idx = []
        for i in range(len(centroids)):
            distances = np.linalg.norm(X - centroids[i], axis=1)
            closest_idx.append(np.argmin(distances))

        
        # # create a scatter plot of the data points
        # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
        # # add the cluster centroids as black crosses
        # plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=3, color='black')
        # # add the closest index points as red dots
        # plt.scatter(X[closest_idx, 0], X[closest_idx, 1], marker='o', s=100, color='red')

        # plt.show()

        return best_k, closest_idx

    

    def distance_matrix(X,y):
        
        dist = DistanceMetric.get_metric('euclidean')
        C1 = dist.pairwise(X)
        C2 = dist.pairwise(y)
        return C1, C2


    def finding_triplets(dist_matrix):
        
        _, closest_idx = kmeans_finder(dist_matrix)
      
        triplets = []
        
        # neigh = NearestNeighbors(n_neighbors=random.randint(2, dist_matrix.shape[0]))
        neigh = NearestNeighbors(n_neighbors=n_neighbors)
        
        neigh.fit(dist_matrix)
        for i in range(len(closest_idx)):
            neigh_dist, neigh_ind = neigh.kneighbors([dist_matrix[closest_idx[i]]])
            j = neigh_ind[0][0]
            k = neigh_ind[0][1]
            l = neigh_ind[0][-1]
            triplets.append((j, k, l))
        return triplets



    
    # define the triplet margin loss function
    triplet_loss_fn = nn.TripletMarginLoss(margin=1)


    def min_gromov_wasserstein_distance1(C1_fixed, C2_fixed, C1, C2, p, q, nb_iter_max = 5, lr = lr, epsilon = epsilon, tol= tol):
    
        loss_iter = []
        
        triplets_1 = finding_triplets(C1_fixed.detach().cpu().numpy())
        
        triplets_2 = finding_triplets(C2_fixed.detach().cpu().numpy())
       
        
        for i in range(nb_iter_max):
            
            print('iter', i)

            loss1 = ot.gromov.entropic_gromov_wasserstein2(C1, C2, p, q, loss_fun="square_loss", epsilon = epsilon, tol=tol, verbose=True, log=False)
            
            loss2 = ot.gromov.entropic_gromov_wasserstein2(C1_fixed, C1, p, p, loss_fun="square_loss", epsilon = epsilon, tol=tol, verbose=True, log=False)

            loss3 = ot.gromov.entropic_gromov_wasserstein2(C2_fixed, C2, q, q, loss_fun="square_loss", epsilon = epsilon, tol=tol, verbose=True, log=False)
        
            loss4 = 0
            loss5 = 0
        
        
            # define the triplet margin loss function
            
            
            for triplet in triplets_1:
                
                anchor1 = C1[triplet[0]]
                pos1 = C1[triplet[1]] 
                neg1 = C1[triplet[2]]
            
                lossC1 = triplet_loss_fn(anchor1, pos1, neg1)
        
                # add the loss to the total loss for this iteration
                loss4 += lossC1
        
    
            for triplet in triplets_2:
                
                anchor2 = C2[triplet[0]] 
                pos2 = C2[triplet[1]] 
                neg2 = C2[triplet[2]]
                    
    
                lossC2 = triplet_loss_fn(anchor2, pos2, neg2)
        
                # add the loss to the total loss for this iteration
                loss5 += lossC2
    
            
            loss = (loss1 + loss2+ loss3) + 1/len(triplets_1)*loss4 + 1/len(triplets_2)*loss5
            #print(loss)
            loss_iter.append(loss.clone().detach().cpu().numpy())
    
        # Compute the gradient of the loss with respect to the cost matrices
                
            loss.backward()
            with torch.no_grad():
                
                grad_C1 = C1.grad
                
                grad_C2 = C2.grad
    
                # print('grad_C1:', grad_C1)
                # print('grad_C2:', grad_C2)
        
                C1 -= lr * grad_C1
                C2 -= lr * grad_C2
            
                
                for m in range(C1.shape[0]):
                    C1[m][m] = 0
             
                for m in range(C2.shape[0]):
                    C2[m][m] = 0

            

                C1.grad.zero_()
            
                C2.grad.zero_()
            
        
        # Convert the final tensors back to numpy arrays
        C1 = C1.detach().cpu().numpy()
        C2 = C2.detach().cpu().numpy()
    
        return C1, C2, loss_iter
    
    X, y  = normalize(X, norm="l2"), normalize(y, norm="l2")
  
    C1, C2 = distance_matrix(X, y)
    
    
    C1_torch = torch.tensor(C1, dtype=torch.float32, requires_grad=True)
    C2_torch = torch.tensor(C2, dtype=torch.float32, requires_grad=True)
  
   
    
    p = ot.unif(C1.shape[0])
    p = torch.tensor(p, dtype=torch.float32)
    q = ot.unif(C2.shape[0])
    q = torch.tensor(q, dtype=torch.float32)
    
    
    C1_opt, C2_opt, loss_iter = min_gromov_wasserstein_distance1(C1_torch, C2_torch, C1_torch, C2_torch, p, q)
    

    plt.figure()
    plt.plot(loss_iter)
    plt.title("Loss along iterations")
    
    time2 = time.time()
    print("model takes {:f}".format(time2-time1), 'seconds')

    return C1, C2, C1_opt, C2_opt


def eval_model1_scot(X,y):
    C1, C2, C1_opt, C2_opt = model(X, y,epsilon =1e-1, tol= 1e-4, lr = 1, n_neighbors = 3)