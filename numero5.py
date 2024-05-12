import math
import numpy
import pandas
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("Рабочая тетрадь номер 5")

num_of_tasks=['1', '2', '3', '4', '5']
data=8

class Trigonoworker:
    def cos_find(self, cos_num):
        self.cos_answer=math.cos(cos_num)
        print(self.cos_answer)
    def sin_find(self, sin_num):
        self.sin_answer=math.sin(sin_num)
        print(self.sin_answer)
    def tg_find(self, tg_num):
        self.tg_answer=math.tan(tg_num)
        print(self.tg_answer)
    def arccos_find(self, arccos_num):
        self.arccos_answer=math.acos(arccos_num)
        print(self.arccos_answer)
    def arcsin_find(self, arcsin_num):
        self.arcsin_answer=math.asin(arcsin_num)
        print(self.arcsin_num)
    def arctg_find(self, arctg_num):
        self.arctg_answer=math.atan(arctg_num)
        print(self.arctg_num)
    def rad_find(self, grad_num):
        self.rad_answer=math.radians(grad_num)
        print(self.rad_answer)

class Tree:
    def __init__(self, data):
        self.left=None
        self.right=None
        self.data=data
    def insert(self, data):
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Tree(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Tree(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data
        self.print_tree()
    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.data)
        if self.right:
            self.right.print_tree()

Trig=Trigonoworker()
Tree_num3=Tree(data)

def task1():
    cos_num=input("Введите x для нахождения косинуса: ")
    try:
        cos_num=int(cos_num)
        Trig.cos_find(cos_num)
    except:
        print("Ошибка в вводе данных")
    sin_num=input("Введите x для нахождения синуса: ")
    try:
        sin_num=int(sin_num)
        Trig.sin_find(sin_num)
    except:
        print("Ошибка в вводе данных")
    tg_num=input("Введите x для нахождения косинуса: ")
    try:
        tg_num=int(tg_num)
        Trig.tg_find(tg_num)
    except:
        print("Ошибка в вводе данных")
    arccos_num=int(input("Введите x для нахождения арккосинуса: "))
    try:
        arccos_num=int(arccos_num)
        Trig.arccos_find(arccos_num)
    except:
        print("Ошибка в вводе данных")
    arcsin_num=input("Введите x для нахождения арксинуса: ")
    try:
        arcsin_num=int(arcsin_num)
        Trig.arcsin_find(arcsin_num)
    except:
        print("Ошибка в вводе данных")
    arctg_num=input("Введите x для нахождения арктангенса: ")
    try:
        arctg_num=int(arctg_num)
        Trig.arctg_find(arctg_num)
    except:
        print("Ошибка в вводе данных")
    grad_num=input("Введите значение в градусах, чтобы перевести его в радианы: ")
    try:
        grad_num=int(grad_num)
        Trig.rad_find(grad_num)
    except:
        print("Ошибка в вводе данных")

def task2():
    tree_num2=[["a", "b", "d"], ["a", "b", "e"], ["a", "c", "f"]]
    print(f"Корень дерева: {tree_num2[1]}")
    print(f"Левая сторона дерева: {tree_num2[0]}")
    print(f"Правая сторона дерева: {tree_num2[2]}")

def task3():
    try:
        data=int(input("Введите значение data: "))
        Tree_num3.insert(data)
    except:
        print("Значение data введено неверно")

def task4():
    x=numpy.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y=[0, 0, 0, 1, 1, 1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    classifier=DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    tree.plot_tree(classifier)
    y_pred=classifier.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    DecisionTreeClassifier()

def task5():
    url = "https://raw.githubusercontent.com/likarajo/petrol_consumption/master/data/petrol_consumption.csv"
    data = pandas.read_csv(url)
    data.head()
    X = data.drop('Petrol_Consumption', axis=1)
    y = data['Petrol_Consumption']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(mse)
    print(r2)

def main():
    print(f"Список заданий: {num_of_tasks}")
    current_task=str(input("Введите номер задания: "))
    if current_task=='1':
        task1()
    if current_task=='2':
        task2()
    if current_task=='3':
        task3()
    if current_task=='4':
        task4()
    if current_task=='5':
        task5()
    if current_task=='0':
        exit()

while True:
    main()