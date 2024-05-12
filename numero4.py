import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

print("Рабочая тетрадь 4")

task_list=["1", "2", "3", "4", "5"]

def task_1():
    print("Первый порядок")
    x=numpy.array([1, 2, 4, 8])
    y=numpy.array([0.1, 0.4, 0.2, 0.5])
    pol_1=numpy.vstack([x, numpy.ones(len(x))]).T
    m, c=numpy.linalg.lstsq(pol_1, y, rcond=None)[0]
    plt.plot(x, y, 'o', label="Исходные данные", markersize=10)
    plt.plot(x, m*x+c, 'r', label='Линейная экстраполяция')
    plt.legend()
    plt.show()
    print(pol_1)
    print()

    print("Второй порядок")
    delta=2.0
    x=numpy.linspace(-5, 5, 7)
    y=x**2+delta*(numpy.random.rand(7)-0.5)
    x+=delta*(numpy.random.rand(7)-0.5)
    x.tofile('x_data.txt', '\n')
    y.tofile('y_data.txt', '\n')
    x=numpy.fromfile('x_data.txt', float, sep='\n')
    y=numpy.fromfile('y_data.txt', float, sep='\n')
    print(x)
    print(y)
    m=numpy.vstack((x**2, x, numpy.ones(7))).T
    s=numpy.linalg.lstsq(m, y, rcond=None)[0]
    x_prec=numpy.linspace(-3, 3, 102)
    plt.plot(x, y, "D")
    plt.plot(x_prec, s[0]*x_prec**2+s[1]*x_prec+s[2], '-', lw=2)
    plt.grid()
    plt.show()
    print()

    print("Третий порядок")
    m=numpy.vstack((x**3, x**2, x, numpy.ones(7))).T
    s=numpy.linalg.lstsq(m, y, rcond=None)[0]
    x_prec=numpy.linspace(-5, 5, 101)
    plt.plot(x, y, "D")
    plt.plot(x_prec, s[0]*x_prec**3+s[1]*x_prec**2+s[2]*x_prec+s[3], '-', lw=3)
    plt.grid()
    plt.show()

def task_2():
    beta=(0.25, 0.75, 0.5)
    def f1(x, b0, b1, b2):
        return b0+b1*numpy.exp(-b2*x**2)
    xdata=numpy.linspace(0, 5, 50)
    y=f1(xdata, *beta)
    ydata=y+0.05*numpy.random.randn(len(xdata))
    beta_opt, beta_cov=sp.optimize.curve_fit(f1, xdata, ydata)
    lin_dev=sum(beta_cov[0])
    residuals=ydata-f1(xdata, *beta_opt)
    fres=sum(residuals**2)
    fig, ax=plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f1(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f1(x, \beta)$", fontsize=18)
    plt.show()

    beta=(0.25, 0.75)
    def f2(x, b0, b1):
        return b0+b1*x
    xdata=numpy.linspace(0, 5, 50)
    y=f2(xdata, *beta)
    ydata=y+0.05*numpy.random.randn(len(xdata))
    beta_opt, beta_cov=sp.optimize.curve_fit(f2, xdata, ydata)
    line_dev=sum(beta_cov[0])
    residuals=ydata-f2(xdata, *beta_opt)
    fres=sum(residuals**2)
    fig, ax=plt.subplots()
    ax.scatter=(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f2(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f2(x, \beta)$", fontsize=18)
    plt.show()

    beta=(0.25, 0.75, 0.5)
    def f3(x, b0, b1, b2):
        return b0+b1*x+b2*x*x
    xdata=numpy.linspace(0, 5, 50)
    y=f3(xdata, *beta)
    ydata=y+0.05*numpy.random.randn(len(xdata))
    beta_opt, beta_cov = sp.optimize.curve_fit(f3, xdata, ydata)
    lin_dev=sum(beta_cov[0])
    residuals=ydata-f3(xdata, *beta_opt)
    fres=sum(residuals**2)
    fig, ax=plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, y, 'r', lw=2)
    ax.plot(xdata, f3(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f3(x, \beta)$", fontsize=18)
    plt.show()

    beta=(1, 2)
    def f4(x, b0, b1):
        return b0+b1*numpy.log(x)
    xdata=numpy.linspace(1, 5, 50)
    y=f4(xdata, *beta)
    ydata=y+0.05*numpy.random.randn(len(xdata))
    beta_opt, beta_cov=sp.optimize.curve_fit(f4, xdata, ydata)
    lin_dev=sum(beta_cov[0])
    residuals=ydata-f4(xdata, *beta_opt)
    fres=sum(residuals**2)
    fig, ax=plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, ydata, 'r', lw=2)
    ax.plot(xdata, f4(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f3(x, \beta)$", fontsize=18)
    plt.show()

    beta=(1, 2)
    def f5(x, b0, b1):
        return b0*x**b1
    xdata=numpy.linspace(1, 5, 50)
    y=f5(xdata, *beta)
    ydata=y+0.5*numpy.random.randn(len(xdata))
    beta_opt, beta_cov=sp.optimize.curve_fit(f5, xdata, ydata)
    lin_dev=sum(beta_cov[0])
    residuals=ydata-f5(xdata, *beta_opt)
    fres=sum(residuals**2)
    fig, ax = plt.subplots()
    ax.scatter(xdata, ydata)
    ax.plot(xdata, ydata, 'r', lw=2)
    ax.plot(xdata, f4(xdata, *beta_opt), 'b', lw=2)
    ax.set_xlim(0, 5)
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$f3(x, \beta)$", fontsize=18)
    plt.show()

def task_3():
    url = "https://raw.githubusercontent.com/AnnaShestova/salary-years-simple-linear-regression/master/Salary_Data.csv"
    data = pandas.read_csv(url)
    print(data.head())
    x = data['YearsExperience'].values.reshape(-1, 1)
    y = data['Salary']
    model = LinearRegression()
    model.fit(x, y)
    intercept = model.intercept_
    slope = model.coef_[0]
    print("Уравнение линейной регрессии: Salary = {:.2f} + {:.2f} * YearsExperience".format(intercept, slope))
    years_of_experience = 10
    predicted_salary = model.predict(numpy.array([[years_of_experience]]))
    print("Прогноз заработной платы для {} лет опыта рабяоты: {:.2f}".format(years_of_experience, predicted_salary[0]))
    plt.scatter(x, y, color='blue')
    plt.plot(x, model.predict(x), color='red')
    plt.title('Заработная плата от опыта работы')
    plt.xlabel('Опыт работы')
    plt.ylabel('Заработная плата')
    plt.show()

def task_4():
    url = "https://raw.githubusercontent.com/likarajo/petrol_consumption/master/data/petrol_consumption.csv"
    data = pandas.read_csv(url)
    print(data.head())
    x = data.drop('Petrol_Consumption', axis=1)
    y = data['Petrol_Consumption']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    coefficients = pandas.DataFrame(model.coef_, x.columns, columns=['Coefficient'])
    print("Коэффициенты множественной линейной регрессии:")
    print(coefficients)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    print("Среднеквадратичная ошибка (MSE):", mse)
    url_wine = "https://raw.githubusercontent.com/aniruddhachoudhury/Red-Wine-Quality/master/winequality-red.csv"
    data_wine = pandas.read_csv(url_wine, sep=';')
    print(data_wine.head())


def task_5():
    var_num=int(input("Введите номер варианта (от 1 до 10) "))

    def polynomial_regression(x, y, degree):
        x_design = numpy.column_stack([numpy.ones_like(x)] + [x ** i for i in range(1, degree + 1)])
        coefficients = numpy.linalg.inv(x_design.T @ x_design) @ x_design.T @ y

        return coefficients

    def predict(x, coefficients):
        y_pred = numpy.dot(x, coefficients)

        return y_pred

    def plot_polynomial(x, y, y_pred, degree):
        plt.scatter(x, y, label='Экспериментальные данные')
        plt.plot(x, y_pred, color='red', label='Полином {} степени'.format(degree))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Аппроксимация полиномом {} степени'.format(degree))
        plt.legend()
        plt.grid(True)
        plt.show()

    if var_num==1:
        x = numpy.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = numpy.array([3.0, 6.0, 3.0, 6.0, 4.0, 3.0])
    if var_num==2:
        x = numpy.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = numpy.array([5.0, 5.0, 4.0, 4.0, 6.0, 6.0])
    if var_num==3:
        x = numpy.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
        y = numpy.array([2.0, 3.0, 3.0, 3.0, 2.0, 4.0])
    if var_num==4:
        x = numpy.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0])
        y = numpy.array([6.0, 2.0, 6.0, 4.0, 3.0, 4.0])
    if var_num==5:
        x = numpy.array([5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
        y = numpy.array([2.0, 4.0, 4.0, 3.0, 3.0, 3.0])
    if var_num==6:
        x = numpy.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
        y = numpy.array([4.0, 3.0, 6.0, 6.0, 4.0, 4.0])
    if var_num==7:
        x = numpy.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        y = numpy.array([2.0, 6.0, 4.0, 4.0, 2.0, 5.0])
    if var_num==8:
        x = numpy.array([5.0, 5.2, 5.4, 5.6, 5.8, 6.0])
        y = numpy.array([3.0, 2.0, 5.0, 2.0, 2.0, 3.0])
    if var_num==9:
        x = numpy.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        y = numpy.array([4.0, 2.0, 4.0, 2.0, 5.0, 2.0])
    if var_num==10:
        x = numpy.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        y = numpy.array([6.0, 3.0, 2.0, 6.0, 2.0, 5.0])
    else:
        print("Номер варианта введен неверно")

    try:
        degree_1 = 1
        degree_2 = 2

        coefficients_1 = polynomial_regression(x, y, degree_1)
        y_pred_1 = predict(numpy.column_stack([numpy.ones_like(x), x]), coefficients_1)

        coefficients_2 = polynomial_regression(x, y, degree_2)
        y_pred_2 = predict(numpy.column_stack([numpy.ones_like(x), x, x ** 2]), coefficients_2)

        print("Коэффициенты полинома первой степени:", coefficients_1)
        print("Коэффициенты полинома второй степени:", coefficients_2)

        table_data = {'X': x, 'Y': x, 'Y_pred_1': y_pred_1, 'Y_pred_2': y_pred_2}
        table = pandas.DataFrame(table_data)
        print(table)

        plot_polynomial(x, y, y_pred_1, degree_1)
        plot_polynomial(x, x, y_pred_2, degree_2)
    except:
        pass

def main():
    while True:
        print(task_list)
        choose_num=int(input("Выберите номер задания (введите 0 для выхода): "))
        if choose_num==0:
            exit()
        if choose_num==1:
            task_1()
        if choose_num==2:
            task_2()
        if choose_num==3:
            task_3()
        if choose_num==4:
            task_4()
        if choose_num==5:
            task_5()

main()