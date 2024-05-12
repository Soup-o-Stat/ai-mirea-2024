import numpy
import random
import pandas
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#created by Soup-o-Stat

debug_mode=0
start=1

#Список заданий
list_of_tasks=["1_3_1", "1_3_2", "1_3_3", "1_3_4", "1_3_5", "1_3_6", "2_3_1", "2_3_2", "2_3_3", "3_3_2"]
current_task=None

print("Кощеев Михаил Ильич")
print()

#Консоль для дебага
#Режим для дебага по умолчанию отключен. чтобы его включить, измените debug_mode с 0 на 1 или впишите в консоль "/debug_mode_1"
#Чтобы вызвать консоль, в поле для ввода номера задания впишите "/console"
def console():
    global debug_mode
    console_run=True
    print("print '/help' for help")
    while console_run==True:
        print("Debug mode ", debug_mode, "\n")
        console_task=str(input())
        if console_task=="/debug_mode_0":
            debug_mode=0
        if console_task=="/debug_mode_1":
            debug_mode=1
        if console_task=="/help":
            print("/help, /debug_mode_1, /debug_mode_0, /exit")
        if console_task=="/exit":
            console_run=False

#Задание 1.3.1
def task_1_3_1():
    a=0
    b=1
    matrix_8x8=numpy.array([[a, b, a, b, a, b, a, b],
                            [b, a, b, a, b, a, b, a],
                            [a, b, a, b, a, b, a, b],
                            [b, a, b, a, b, a, b, a],
                            [a, b, a, b, a, b, a, b],
                            [b, a, b, a, b, a, b, a],
                            [a, b, a, b, a, b, a, b],
                            [b, a, b, a, b, a, b, a]])
    print(matrix_8x8)

#Задание 1.3.2
def task_1_3_2():
    matrix_5x5 = numpy.tile(numpy.arange(5), (5, 1))
    print(matrix_5x5)

#Задание 1.3.3
def task_1_3_3():
    array = numpy.random.rand(3, 3, 3)
    print(array)

#Задание 1.3.4
def task_1_3_4():
    matrix=numpy.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]])
    print(matrix)

#Задание 1.3.5
def task_1_3_5():
    array = [random.randint(1, 100) for i in range(10)]
    array.sort(reverse=True)
    print(array)

#Задание 1.3.6
def task_1_3_6():
    matrix = numpy.random.randint(1, 134, size=(5, 5))
    print(matrix)
    print("Форма матрицы:", matrix.shape)
    print("Размер матрицы:", matrix.size)
    print("Размерность матрицы:", matrix.ndim)

#Задание 2.3.1
def task_2_3_1():
    a = numpy.array(random.randint(1, 10))
    b = numpy.array(random.randint(1, 10))
    dist = numpy.sqrt(numpy.sum((a - b) ** 2))
    print("a = ", a)
    print("b = ", b)
    print(f"Евклидово расстояние равно {dist}")

#Задание 2.3.2
def task_2_3_2():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    frame = pandas.read_csv(url)
    print(frame.head())

#Задание 2.3.3
def task_2_3_3():
    url='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    dataframe=pandas.read_csv(url)
    print(dataframe.head(5), dataframe.tail(3), dataframe.shape, dataframe.describe())
    print(dataframe.iloc[1:4])
    print(dataframe[dataframe["Sex"]=="male"].head(3))

#Задание 3.3.2
def task_3_3_2():
    url="https://raw.githubusercontent.com/akmand/datasets/master/iris.csv"
    data = pandas.read_csv(url)
    data['sepal_length_cm'] = MinMaxScaler().fit_transform(data['sepal_length_cm'].values.reshape(-1, 1))
    data['sepal_width_cm'] = StandardScaler().fit_transform(data['sepal_width_cm'].values.reshape(-1, 1))
    print(data.head())

#Функция-вызов функции определенного задания
def task_choose():
    global start
    if current_task == "1_3_1":
        task_1_3_1()
    if current_task=="1_3_2":
        task_1_3_2()
    if current_task=="1_3_3":
        task_1_3_3()
    if current_task=="1_3_4":
        task_1_3_4()
    if current_task=="1_3_5":
        task_1_3_5()
    if current_task=="1_3_6":
        task_1_3_6()
    if current_task=="2_3_1":
        task_2_3_1()
    if current_task=="2_3_2":
        task_2_3_2()
    if current_task=="2_3_3":
        task_2_3_3()
    if current_task=="3_3_2":
        task_3_3_2()
    if current_task=="exit":
        start=False

#Ядро кода. Без ядра код не работает, не вынимайте ядро
#Для выхода из программы напишите в поле для ввода номера задания "exit"
def main():
    global current_task
    while start==True:
        print("Номера заданий:")
        print(list_of_tasks)
        current_task=input("Введите номер задания (для выхода напишите exit): ")
        if current_task=="/console":
            console()
        if debug_mode==0:
            try:
                task_choose()
            except:
                #чтобы программа умирала от ошибок в коде (которых быть не должно) измените параметр debug_mode на 1 (или True)
                #или измените его значение в консоли (для этого, когда вам предложат ввести номер задания введите "/console")
                print("Где-то в коде спряталась ошибка (^ ω ^)") #умные дяденьки из Valkyrie Initiative посоветовали мне писать user friendly код
        else:
            task_choose()
main()