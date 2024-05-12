import random
import numpy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn
from sklearn.metrics import accuracy_score

task_list=["1_3_1", "2_3_1", "3_2_2"]

start=True

print("Рабочая тетрадь номер 3")
print()

#Задание 1_3_1
#Задаются 4 рандомные точки в пространстве, рассчитывается расстояние между ними (Евклида, квадрат Евклида, Чебышева)
def task_1_3_1():
    x_list=[]
    y_list=[]
    z_list=[]
    fig=plt.figure()
    ax=fig.add_subplot(111, projection="3d")
    for i in range(1, 5):
        x_list.append(random.randint(1,10))
        y_list.append(random.randint(1,10))
        z_list.append(random.randint(1,10))

    ax.scatter(x_list[0], y_list[0], z_list[0])
    ax.scatter(x_list[1], y_list[1], z_list[1])
    ax.scatter(x_list[2], y_list[2], z_list[2])
    ax.scatter(x_list[3], y_list[3], z_list[3])

    print("Расстояние Евклида")
    cords1=[x_list[0], y_list[0], z_list[0]]
    cords2 = [x_list[1], y_list[1], z_list[1]]
    cords3 = [x_list[2], y_list[2], z_list[2]]
    cords4 = [x_list[3], y_list[3], z_list[3]]
    points = np.array([cords1, cords2, cords3, cords4])
    distances = numpy.zeros((4, 4))
    for i in range(4):
        for j in range(i + 1, 4):
            distances[i, j] = numpy.linalg.norm(points[i] - points[j])
            distances[j, i] = distances[i, j]
    print(distances)
    print("Квадрат расстояния Евклида")
    print(distances**2)
    print("Расстояние Чебышева")
    distance_chebyshev = np.max(np.abs(points[:, np.newaxis, :] - points), axis=-1)
    print(distance_chebyshev)
    plt.show()

#Задание 2_3_1
#Модифицированный код из примера. Задаются ближайшие соседи и размер тестовой выборки в процентах (все задается пользователем)
def task_2_3_1():
    iris=seaborn.load_dataset('iris')
    print(iris)
    plt.figure(figsize=(16, 7))

    plt.subplot(121)
    seaborn.scatterplot(
        data=iris,
        x='petal_width',
        y='petal_length',
        hue='species',
        s=70)
    plt.xlabel('Длина лепестка, см')
    plt.ylabel('Ширина лепесткаб см')
    plt.legend()
    plt.grid()

    plt.subplot(122)
    seaborn.scatterplot(
        data=iris,
        x='petal_width',
        y='petal_length',
        hue='species',
        s=70)
    plt.xlabel('Длина лепестка, см')
    plt.ylabel('Ширина лепесткаб см')
    plt.legend()
    plt.grid()

    x_train, x_test, y_train, y_test = train_test_split(
        iris.iloc[:, :-1],
        iris.iloc[:, -1],
        test_size=0.2)
    x_train.shape, x_test.shape, y_train.shape, y_test.shape
    x_train.head()
    y_train.head()

    k=int(input("Выберите кол-во ближайших соседей: "))
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    y_pred=model.predict(x_test)
    print(f"Предсказание модели: {y_pred}")
    perc=int(input("Выберите, какой процент считать моей никчемной программе: "))
    plt.figure(figsize=(10, 7))
    seaborn.scatterplot(x='petal_width', y='petal_length', data=iris, hue='species', s=70)
    plt.xlabel('Длина лепестка, см')
    plt.ylabel('Ширина лепесткаб см')
    plt.legend(loc=2)
    plt.grid()
    plt.show()

    for i in range((len(y_test))*perc//100):
        if numpy.array(y_test)[i]!=y_pred[i]:
            plt.scatter(x_test.iloc[i, 3], x_test.iloc[i, 2], color='red', s=150)
    print(f"accuracy: {accuracy_score(y_test, y_pred):.3}")

# Задание 3_2_2
#Определение наборов признаков человека
def task_3_2_2():
    person_features = \
    {
        'цвет глаз': 'голубой',
        'цвет волос': 'черный',
        'рост': 183,
        'вес': 60,
        'цена': 300
    }
    person_features_matrix = np.array([
        [person_features['цвет глаз']],
        [person_features['цвет волос']],
        [person_features['рост']],
        [person_features['вес']],
        [person_features['цена']]
    ])
    print(person_features_matrix)

def main():
    global start, task_list, start_1_3_1
    while start:
        print(task_list)
        current_task=str(input("Введите номер задания (для выхода введите exit): "))
        if current_task=="1_3_1":
            task_1_3_1()
        if current_task=="2_3_1":
            task_2_3_1()
        if current_task=="3_2_2":
            task_3_2_2()
        if current_task=="exit" or current_task=="Exit" or current_task=="выход" or current_task=="Выход":
            exit()

main()