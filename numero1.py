import math
import numpy
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numpy import trapz
import yfinance as yf

print("Рабочая тетрадь 1")

list_of_tasks=["1_3", "2_3", "3_3_1", "3_3_2", "3_3_3", "3_3_4", "3_3_5", "4_3_1", "4_3_2", "4_3_3", "4_3_4", "4_3_5"]

#Задание 1.3: вывод на печать и определение типа переменной
def num1_3():
    x=5>=2
    A={1, 3, 7, 8}
    B={2, 4, 5, 10, 'apple'}
    C=A&B
    df='Антонова Антонина', 34, 'ж'
    z='type'
    D=[1, 'title', 2, 'content']
    print(x, "|" , type(x), "\n",
           A, "|", type(A), "\n",
           B, "|", type(B), "\n",
           C, "|", type(C), "\n",
           df, "|", type(df), "\n",
           z, "|", type(z), "\n",
           D, "|", type(D))

#Задание 2.3: задается х, напечатать какому из интервалов принадлежит x
def num2_3():
    start=True
    while start==True:
        try:
            x=int(input("Введите значение x: "))
            start=False
        except:
            print("Введено некорректное значение")
    if start==False:
        if x<-5:
            print("x принадлежит интервалу (∞, -5)")
        if x>=-5 and x<=5:
            print("x принадлежит интервалу [-5, 5]")
        if x>5:
            print("x принадлежит интервалу (5, ∞)")

#Задание 3.3.1: вывод числа из примера на while (3.2.1) в обратном порядке.
def num3_3_1():
    x=10
    while x>=1:
        print(x)
        x-=3

#Задание 3.3.2: создание списка значимых характеристик (признаков), идентифицирующих человека, вывод списка на экран
def num3_3_2():
    list_of_skills=["Атлетичный", "Храбрый", "Шустрый", "Книголюб", "В форме", "Грациозный", "Агорафобия", "Неуклюжий"]
    print(list_of_skills)

#Задание 3.3.3: создание списка чисел от 2 до 15 с шагом 1
def num3_3_3():
    x=1
    list_of_pieces_of_nums=[]
    while x!=15:
        x+=1
        list_of_pieces_of_nums.append(x)
    print(list_of_pieces_of_nums)

#Задание 3.3.4: вывод числа из примера на for с функцией range() (3.2.4) в обратном порядке
def num3_3_4():
    for i in range(105, 4, -25):
        print(i)

#Задание 3.3.5: переставление элементов массива x с четными индексами в обратном порядке
def num3_3_5():
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    even_elements = [x[i] for i in range(0, len(x), 2)]
    even_elements.reverse()
    x[::2] = even_elements
    print(x)

#Задание 4.3.1: создание массива случайных значений из интервала (0;1), расчет средних и медиаданных значений для
#массива, сравнение результатов
def num4_3_1():
    n=int(input("Введите кол-во n: "))
    x = numpy.random.random(n)
    sredn = numpy.mean(x)
    median = numpy.median(x)
    print('Среднее значение:', sredn)
    print('Медианное значение:', median)
    if sredn == median:
        print('Среднее и медианное значения совпадают.')
    else:
        print('Среднее и медианное значения не совпадают.')
    plt.scatter(range(n), x)
    plt.xlabel('Позиция')
    plt.ylabel('Значение')
    plt.show()

#Задание 4.3.2: создание массива из 10 значений функции, выделение срез первой половины массива и построение график
#для основного массива (линейный) и для среза (точечный)

def function_4_3_2(x):
    return ((math.sqrt(1+math.e**math.sqrt(x)+math.cos(x**2))))/math.fabs(1 - math.sin(x) ** 3)

def num4_3_2():
    x = numpy.linspace(1, 10, 10)
    y = numpy.array([function_4_3_2(xi) for xi in x])
    x_slice = x[:5]
    y_slice = y[:5]
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Основной массив', color='blue')
    plt.scatter(x_slice, y_slice, label='Срез', color='red')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('График функции и его срез')
    plt.legend()
    plt.grid(True)
    plt.show()

#Задание 4.3.3: построение графика на интервале (0,10) с шагом 1 с заливкой площади и нахожденик эту площади под ним
def function_4_3_3(x):
    return numpy.abs(numpy.cos(x*numpy.exp(numpy.cos(x) + numpy.log(x+1))))

def num4_3_3():
    x = numpy.arange(0, 10, 1)
    y = function_4_3_3(x)
    area_simps = simps(y, x)
    print(f'Площадь под кривой (simps): {area_simps}')

    plt.fill_between(x, y, color="skyblue", alpha=0.4)
    plt.plot(x, y, color="blue")
    plt.xlabel('x')
    plt.ylabel('|cos(xe^(cos(x)+ln(x+1)))|')
    plt.title('График функции |cos(xe^(cos(x)+ln(x+1)))|')
    plt.show()

#Задание 4.3.4: построение 3 график на плоскости и оценка динамик акций Apple, Microsoft и Google за 2021 год
def num4_3_4():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    stocks_data = yf.download(tickers, start='2021-01-01', end='2021-12-31')['Close']

    fig, axs = plt.subplots(3, figsize=(12, 10))
    for i, ticker in enumerate(tickers):
        axs[i].plot(stocks_data[ticker], label=ticker)
        axs[i].set_title(f'Stock Price of {ticker}')
        axs[i].legend()
    plt.tight_layout()
    plt.show()

#Задание 4.3.5: простейший калькуляор включающий основные действия для двух переменных, а также вычисление функций
def num4_3_5():
    try:
        x=int(input("Введите первое число (x): "))
        y=int(input("Введите второе число (y): "))
        print("Что вы хотите сделать с этими цифрами?")
        print(" 1- сложить (+)", "\n", "2 - вычесть (-)", "\n", "3 - умножить (*)", "\n", "4 - разделить (/)", "\n",
              "5 - другое")
        operation=str(input())
        if operation=="5" or operation=="другое" or operation=="Другое":
            print("Выберите функцию:", "\n", "1 - e⁽ˣ₊ʸ⁾", "\n", "2 - sin(x+y)", "\n", "3 - cos(x+y)", "\n", "4 - xʸ")
            operation_xxx=int(input())
            if operation_xxx==1:
                answer=math.e**(x+y)
            if operation_xxx==2:
                answer=math.sin(x+y)
            if  operation_xxx==3:
                answer=math.cos(x+y)
            if operation_xxx==4:
                answer=x**y
        if operation=="1" or operation=="Сложить" or operation=="сложить":
            answer=x+y
        if operation=="2" or operation=="вычесть" or operation=="Вычесть":
            answer=x-y
        if operation=="3" or operation=="умножить" or operation=="Умножить":
            answer=x*y
        if operation=="4" or operation=="Разделить" or operation=="умножить":
            if y==0:
                print("Ошибка!!! y не должен быть равен 0! (Делить на 0 нельзя)")
            else:
                answer=x/y
        try:
            print(f"Ответ: {answer}")
        except:
            pass
    except:
        print("Ошибка в вводе данных")

#запуск программы
def main():
    print("Доступные задания:", "\n", list_of_tasks)
    num=str(input("Введите номер задания: "))
    if num=="1_3":
        print()
        num1_3()
    if num=="2_3":
        print()
        num2_3()
    if num=="3_3_1":
        print()
        num3_3_1()
    if num=="3_3_2":
        print()
        num3_3_2()
    if num=="3_3_3":
        print()
        num3_3_3()
    if num=="3_3_4":
        print()
        num3_3_4()
    if num=="3_3_5":
        print()
        num3_3_5()
    if num=="4_3_1":
        print()
        num4_3_1()
    if num=="4_3_2":
        print()
        num4_3_2()
    if num=="4_3_3":
        print()
        num4_3_3()
    if num=="4_3_4":
        print()
        num4_3_4()
    if num=="4_3_5":
        print()
        num4_3_5()

main()