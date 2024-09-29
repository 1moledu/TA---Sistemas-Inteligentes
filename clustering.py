import sys
import os
import random
import math
from map import Map
from shared_instances import SharedInstances


def euclidean_distance(x1, x2):
    return ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) ** 0.5

# Função para inicializar os centróides aleatoriamente
def initialize_centroids(X, k):
    # Escolher os k primeiros pontos como centróides iniciais
    return X[:k]

# Função para atribuir cada ponto ao centróide mais próximo
def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster = distances.index(min(distances))
        clusters.append(cluster)
    return clusters

# Função para atualizar os centróides
def update_centroids(X, clusters, k):
    centroids = []
    for cluster in range(k):
        cluster_points = [X[i] for i in range(len(X)) if clusters[i] == cluster]
        if cluster_points:
            centroid = [sum(p[dim] for p in cluster_points) / len(cluster_points) for dim in range(len(cluster_points[0]))]
            centroids.append(centroid)
    return centroids

# Função principal do algoritmo K-Means
def kmeans(X, k, max_iterations=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iterations):
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if centroids == new_centroids:
            break
        centroids = new_centroids
    return clusters, centroids

# Sua matriz de dados X
def vetorVitimas(mapa):
    possicaoVitima = []

    min_x = min(key[0] for key in mapa.map_data.keys())
    max_x = max(key[0] for key in mapa.map_data.keys())
    min_y = min(key[1] for key in mapa.map_data.keys())
    max_y = max(key[1] for key in mapa.map_data.keys())

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            item = (mapa.get((x, y)))
            if item:
                if item[1] != -1:
                    possicaoVitima.append((x, y))
                    
    
    return(possicaoVitima)

def dictVitimas():
    
    #print(SharedInstances.exp.victims | SharedInstances.exp1.victims | SharedInstances.exp2.victims | SharedInstances.exp3.victims)
    return dict.items(SharedInstances.exp.victims | SharedInstances.exp1.victims | SharedInstances.exp2.victims | SharedInstances.exp3.victims)

def mergeMapas():
    if SharedInstances.exp.retornou and SharedInstances.exp1.retornou and SharedInstances.exp2.retornou and SharedInstances.exp3.retornou:
       
        merged_map = Map()
        dict1 = SharedInstances.exp.map
        dict2 = SharedInstances.exp1.map
        dict3 = SharedInstances.exp2.map
        dict4 = SharedInstances.exp3.map

        mapaTXT(dict1)

        merged_map = merge(dict1, dict2)
            
        merged_map = merge(merged_map, dict3)
    
        #print("linha")
        merged_map = merge(merged_map, dict4)
        #merged_map.draw()
        #print("linha")
        return (True, merged_map)
    else:
        return (False, None)

def merge(dict1, dict2):

    min_x1 = min(key[0] for key in dict1.map_data.keys())
    max_x1 = max(key[0] for key in dict1.map_data.keys())
    min_y1 = min(key[1] for key in dict1.map_data.keys())
    max_y1 = max(key[1] for key in dict1.map_data.keys())

    min_x2 = min(key[0] for key in dict2.map_data.keys())
    max_x2 = max(key[0] for key in dict2.map_data.keys())
    min_y2 = min(key[1] for key in dict2.map_data.keys())
    max_y2 = max(key[1] for key in dict2.map_data.keys())


    if min_x1 < min_x2:
        min_x = min_x1
    else:
        min_x = min_x2
    
    if min_y1 < min_y2:
        min_y = min_y1
    else:
        min_y = min_y2
    
    if max_x1 > max_x2:
        max_x = max_x1
    else:
        max_x = max_x2
    
    if max_y1 > max_y2:
        max_y = max_y1
    else:
        max_y = max_y2
    
    mapRetorno = Map()

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            item = (dict1.get((x, y)))
            item2 = (dict2.get((x, y)))
            if item is not None:   
                mapRetorno.add(((x, y)), item[0], item[1], item[2])
            if item2 is not None:
                mapRetorno.add(((x, y)), item2[0], item2[1], item2[2])
            

    return mapRetorno

def mapaTXT(mapa):

    with open("MapaTXTw", 'w') as arquivo:
    # Escrever no arquivo
        linha = ""
        min_x = min(key[0] for key in mapa.map_data.keys())
        max_x = max(key[0] for key in mapa.map_data.keys())
        min_y = min(key[1] for key in mapa.map_data.keys())
        max_y = max(key[1] for key in mapa.map_data.keys())

        for y in range(min_y, max_y + 1):
            arquivo.write(linha + "\n")
            linha = ""
            for x in range(min_x, max_x + 1):
                item = mapa.get((x, y))
                if item:
                    if item[1] == -1:
                        linha += f"[{item[0]:7.2f}  no] " 
                    else:
                        linha += f"[{item[0]:7.2f} {item[1]:3d}] " 
                else:
                    linha +=   f"[     ?     ] " 