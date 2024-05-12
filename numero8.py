import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs
from sklearn.datasets import load_iris
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

print("Рабочая тетрадь номер 8")

task_list=["1", "2"]

def task1():
    X = numpy.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 45], [85, 70], [71, 80], [60, 78], [55, 52], [80, 91], ])
    plt.scatter(X[:, 0], X[:, 1], s=20)
    print(plt.show())
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
    plt.show()
    dataframe = load_iris()
    X = dataframe.data
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(dataframe.data)
    y_kmeans = kmeans.predict(dataframe.data)
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=20, cmap='summer')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='blue', s=100, alpha=0.9)
    plt.show()

def task2():
    url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
    customer_data = pd.read_csv(url)
    customer_data.head()
    print(customer_data.shape)
    data = customer_data.iloc[:, 2:4].values
    plt.figure(figsize=(28, 12), dpi=180)
    plt.figure(figsize=(10, 7))
    dend = shc.dendrogram(shc.linkage(data, method='ward'))
    plt.show()
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(data)
    plt.figure(figsize=(10, 7))
    plt.scatter(data[:, 0], data[:, 1], c=cluster.labels_, cmap='rainbow')

def main():
    print()
    print("Список заданий:")
    print(task_list)
    current_task=str(input("Введите номер задания: "))
    if current_task=="0":
        exit()
    if current_task=="1":
        task1()
    if current_task=="2":
        task2()

while True:
    main()