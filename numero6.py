import random
import networkx as nx
import math

print("Рабочая тетрадб номер 6")
print()

task_list=["1", "2", "3"]

num_1_cycle=1
var_1=1
var_2=1

def qZ(x, y):
    global var_1
    if var_1==1:
        return (x - 3 * y + 1) / (3 * x ** 2 + y ** 2 + 1)
    if var_1==2:
        return (x-2*y-3)/(x**2+3*y**2+1)
    if var_1==3:
        return (x-3*y-2)/(x**2+y**2+1)
    if var_1==4:
        return (x+3*y)/(3*x**2+y**2+1)
    if var_1==5:
        return (x-3*y+1)/(3*x**2+y**2+1)
    if var_1==6:
        return (x+3*y)/(x**2+y**2+1)
    if var_1==7:
        return (x+3*y-3)/(3*x**2+y**2+1)
    if var_1==8:
        return (x-3*y-3)/(x**2+2*y**2+1)
    if var_1==9:
        return (x-2*y)/(x**2+y**2+1)
    if var_1==10:
        return (x-3*y)/(2*x**2+2*y**2+1)

def qSumZ(Z):
    return sum(Z)

def exchangeScheme(oldX, oldY, sortedId):
    x = [0 for i in range(4)]
    y = [0 for i in range(4)]
    x[2] = oldX[sortedId[2]]
    x[3] = oldX[sortedId[2]]
    x[0] = oldX[sortedId[0]]
    x[1] = oldX[sortedId[1]]
    y[0] = oldY[sortedId[2]]
    y[1] = oldY[sortedId[2]]
    y[2] = oldY[sortedId[0]]
    y[3] = oldY[sortedId[1]]
    return x, y

def sorting(Z):
    sortedId = sorted(range(len(Z)), key=lambda k: Z[k])
    return sortedId

def evoStep(x, y, z):
    _, minId = min((value, id) for (id, value) in enumerate(z))
    x = x[:]
    y = y[:]
    z = z[:]
    x.pop(minId)
    y.pop(minId)
    z.pop(minId)
    return x, y, z

def evoSteps(x, y, steps_num=4):
    results = []
    for i in range(steps_num):
        arrZ = [qZ(k, y[i]) for i, k in enumerate(x)]
        x, y, z = evoStep(x, y, arrZ)
        x, y = exchangeScheme(x, y, sorting(z))
        results.append([x, y, qSumZ(arrZ), arrZ])
    return x, y, results

def task_1():
    global var_1
    while var_1!=0:
        var_1=int(input("Введите номер варианта: "))
        if var_1==1:
            x = [-2, -1, 0, 2]
            y = [-2, 0, -1, 1]
        if var_1==2:
            x=[-4, -2, 0, 2]
            y=[-1, 1, 0, -2]
        if var_1==3:
            x=[-1, 0, 2, 3]
            y=[-2, 1, 0, -1]
        if var_1==4:
            x=[-1, 0, 2, 4]
            y=[-2, 1, -1, 0]
        if var_1==5:
            x=[-2, -1, 0, 2]
            y=[-2, 0, -1, 1]
        if var_1==6:
            x=[-5, -3, -2, -1]
            y=[-1, -2, 0, 1]
        if var_1==7:
            x=[-5, -3, -2, 0]
            y=[-1, -2, 0, 1]
        if var_1==8:
            x=[-5, -3, -2, -1]
            y=[-1, -2, 0, 1]
        if var_1==9:
            x=[-1, 0, 2, 3]
            y=[0, -1, -2, 1]
        if var_1==10:
            x=[-1, 0, 2, 3]
            y=[0, 1, -2, 2]
        if var_1==0:
            continue
        results = evoSteps(x, y)
        for i in range(len(results[2])):
            print(f'max_{i + 1}_step: {results[2][i][2]}')
        qualityArrZ = []
        for i in range(len(results[2])):
            qualityArrZ += results[2][i][3]
        print(f'max Z: {max(qualityArrZ)}')

def task_2():
    global var_2
    while var_2!=0:
        var_2=int(input("Введите номер варианта: "))
        if var_2==1:
            distances = [(1, 2, 26),
                         (1, 3, 42),
                         (1, 4, 44),
                         (1, 5, 31),
                         (1, 6, 24),
                         (2, 3, 20),
                         (2, 4, 34),
                         (2, 5, 40),
                         (2, 6, 15),
                         (3, 4, 23),
                         (3, 5, 43),
                         (3, 6, 20),
                         (4, 5, 27),
                         (4, 6, 22),
                         (5, 6, 26)]

            V = [1, 2, 3, 4, 5, 6, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 6),
                 (6, 2)]
            P = [90, 45, 43, 31]
        if var_2==2:
            distances = [(1, 2, 25),
                         (1, 3, 41),
                         (1, 4, 38),
                         (1, 5, 27),
                         (1, 6, 20),
                         (2, 3, 21),
                         (2, 4, 34),
                         (2, 5, 39),
                         (2, 6, 17),
                         (3, 4, 24),
                         (3, 5, 40),
                         (3, 6, 22),
                         (4, 5, 21),
                         (4, 6, 21),
                         (5, 6, 22)]

            V = [1, 3, 5, 4, 6, 2, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 6),
                 (6, 2)]
            P = [41, 60, 85, 60]
        if var_2==3:
            distances = [(1, 2, 23),
                         (1, 3, 42),
                         (1, 4, 40),
                         (1, 5, 25),
                         (1, 6, 22),
                         (2, 3, 20),
                         (2, 4, 30),
                         (2, 5, 34),
                         (2, 6, 13),
                         (3, 4, 22),
                         (3, 5, 41),
                         (3, 6, 21),
                         (4, 5, 26),
                         (4, 6, 19),
                         (5, 6, 22)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(4, 5),
                 (5, 6),
                 (2, 4),
                 (6, 2)]
            P = [78, 24, 63, 17]
        if var_2==4:
            distances = [(1, 2, 17),
                         (1, 3, 39),
                         (1, 4, 32),
                         (1, 5, 28),
                         (1, 6, 18),
                         (2, 3, 24),
                         (2, 4, 28),
                         (2, 5, 35),
                         (2, 6, 13),
                         (3, 4, 25),
                         (3, 5, 43),
                         (3, 6, 23),
                         (4, 5, 20),
                         (4, 6, 16),
                         (5, 6, 24)]

            V = [1, 5, 2, 6, 3, 4, 1]
            Z = [(3, 4),
                 (4, 5),
                 (5, 2),
                 (6, 2)]
            P = [78, 79, 25, 82]
        if var_2==5:
            distances = [(1, 2, 18),
                         (1, 3, 41),
                         (1, 4, 36),
                         (1, 5, 29),
                         (1, 6, 19),
                         (2, 3, 27),
                         (2, 4, 31),
                         (2, 5, 37),
                         (2, 6, 15),
                         (3, 4, 19),
                         (3, 5, 42),
                         (3, 6, 23),
                         (4, 5, 24),
                         (4, 6, 17),
                         (5, 6, 24)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(2, 4),
                 (3, 4),
                 (4, 6),
                 (5, 6)]
            P = [63, 49, 45, 53]
        if var_2==6:
            distances = [(1, 2, 22),
                         (1, 3, 43),
                         (1, 4, 39),
                         (1, 5, 28),
                         (1, 6, 20),
                         (2, 3, 26),
                         (2, 4, 33),
                         (2, 5, 36),
                         (2, 6, 17),
                         (3, 4, 22),
                         (3, 5, 40),
                         (3, 6, 24),
                         (4, 5, 22),
                         (4, 6, 19),
                         (5, 6, 20)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(2, 4),
                 (4, 6),
                 (3, 5),
                 (5, 2)]
            P = [51, 23, 29, 31]
        if var_2==7:
            distances = [(1, 2, 24),
                         (1, 3, 41),
                         (1, 4, 36),
                         (1, 5, 22),
                         (1, 6, 19),
                         (2, 3, 21),
                         (2, 4, 33),
                         (2, 5, 33),
                         (2, 6, 14),
                         (3, 4, 27),
                         (3, 5, 39),
                         (3, 6, 23),
                         (4, 5, 20),
                         (4, 6, 20),
                         (5, 6, 19)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 2),
                 (6, 2)]
            P = [33, 82, 51, 76]
        if var_2==8:
            distances = [(1, 2, 19),
                         (1, 3, 39),
                         (1, 4, 35),
                         (1, 5, 26),
                         (1, 6, 18),
                         (2, 3, 26),
                         (2, 4, 33),
                         (2, 5, 37),
                         (2, 6, 14),
                         (3, 4, 22),
                         (3, 5, 41),
                         (3, 6, 21),
                         (4, 5, 22),
                         (4, 6, 19),
                         (5, 6, 24)]

            V = [1, 4, 2, 3, 5, 6, 1]
            Z = [(5, 1),
                 (4, 5),
                 (2, 3),
                 (3, 4)]
            P = [88, 54, 24, 64]
        if var_2==0:
            continue
        T = 100

        def probability(delta, T):
            return 100 * math.e ** (-delta / T)

        def reductTemp(prevT):
            nextT = 0.5 * prevT
            return nextT

        def edgeLength(i, j, distances, roundTrip=True):
            if roundTrip:
                return max([(item[2] if (item[0] == i and item[1] == j) or (item[1] == i and item[0] == j) else -1)
                            for item in distances])
            else:
                return max([(item[2] if (item[0] == i and item[1] == j) else -1) for item in distances])

        def routeLength(V, distances):
            edges = []
            for i in range(len(V) - 1):
                edges.append(edgeLength(V[i], V[i + 1], distances))
            return sum(edges)

        def routeOneReplacement(arrV, Z, replacementByName=True):
            decrement = 1 if replacementByName else 0
            arrV[Z[0] - decrement], arrV[Z[1] - decrement] = arrV[Z[1] - decrement], arrV[Z[0] - decrement]
            return arrV

        def routeReplacement(V, Z):
            for z in Z:
                V = routeOneReplacement(V, z)
            return V

        def chooseRoute(distances, V, Z, T, P):
            sumLength = routeLength(V, distances)
            arrSum = [sumLength]
            for i in range(len(Z)):
                newV = routeOneReplacement(V[:], Z[i])
                newS = routeLength(newV, distances)
                arrSum.append(newS)
                deltaS = newS - sumLength
                if deltaS > 0:
                    p = probability(deltaS, T)
                    if p > P[i]:
                        V = newV
                        sumLength = newS
                else:
                    V = newV
                    sumLength = newS
                T = reductTemp(T)
            return V, arrSum

        def drawRouteGraph(distances, bestRoute):
            newDistances = []
            for i in range(len(bestRoute) - 1):
                for distance in distances:
                    if distance[0] == bestRoute[i] and distance[1] == bestRoute[i + 1] or distance[1] == bestRoute[i] and \
                            distance[0] == bestRoute[i + 1]:
                        newDistances.append(distance)
            graph = nx.Graph()
            graph.add_weighted_edges_from(newDistances)
            nx.draw_kamada_kawai(graph, node_color='#fb7258', node_size=2000, with_labels=True)
            print("done")

        bestRoute, arrLength = chooseRoute(distances, V, Z, T, P)

        print(f'Лучший выбранный маршрут: {bestRoute}')
        print(f'Длина лучшего выбранного маршрута: {routeLength(bestRoute, distances)}')
        print(f'Длины всех рассмотренных маршрутов: {arrLength}')

def task_3():
    global var_2
    while var_2!=0:
        var_2=int(input("Введите номер варианта: "))
        if var_2==1:
            distances = [(1, 2, 26),
                         (1, 3, 42),
                         (1, 4, 44),
                         (1, 5, 31),
                         (1, 6, 24),
                         (2, 3, 20),
                         (2, 4, 34),
                         (2, 5, 40),
                         (2, 6, 15),
                         (3, 4, 23),
                         (3, 5, 43),
                         (3, 6, 20),
                         (4, 5, 27),
                         (4, 6, 22),
                         (5, 6, 26)]

            V = [1, 2, 3, 4, 5, 6, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 6),
                 (6, 2)]
            P = [90, 45, 43, 31]
        if var_2==2:
            distances = [(1, 2, 25),
                         (1, 3, 41),
                         (1, 4, 38),
                         (1, 5, 27),
                         (1, 6, 20),
                         (2, 3, 21),
                         (2, 4, 34),
                         (2, 5, 39),
                         (2, 6, 17),
                         (3, 4, 24),
                         (3, 5, 40),
                         (3, 6, 22),
                         (4, 5, 21),
                         (4, 6, 21),
                         (5, 6, 22)]

            V = [1, 3, 5, 4, 6, 2, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 6),
                 (6, 2)]
            P = [41, 60, 85, 60]
        if var_2==3:
            distances = [(1, 2, 23),
                         (1, 3, 42),
                         (1, 4, 40),
                         (1, 5, 25),
                         (1, 6, 22),
                         (2, 3, 20),
                         (2, 4, 30),
                         (2, 5, 34),
                         (2, 6, 13),
                         (3, 4, 22),
                         (3, 5, 41),
                         (3, 6, 21),
                         (4, 5, 26),
                         (4, 6, 19),
                         (5, 6, 22)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(4, 5),
                 (5, 6),
                 (2, 4),
                 (6, 2)]
            P = [78, 24, 63, 17]
        if var_2==4:
            distances = [(1, 2, 17),
                         (1, 3, 39),
                         (1, 4, 32),
                         (1, 5, 28),
                         (1, 6, 18),
                         (2, 3, 24),
                         (2, 4, 28),
                         (2, 5, 35),
                         (2, 6, 13),
                         (3, 4, 25),
                         (3, 5, 43),
                         (3, 6, 23),
                         (4, 5, 20),
                         (4, 6, 16),
                         (5, 6, 24)]

            V = [1, 5, 2, 6, 3, 4, 1]
            Z = [(3, 4),
                 (4, 5),
                 (5, 2),
                 (6, 2)]
            P = [78, 79, 25, 82]
        if var_2==5:
            distances = [(1, 2, 18),
                         (1, 3, 41),
                         (1, 4, 36),
                         (1, 5, 29),
                         (1, 6, 19),
                         (2, 3, 27),
                         (2, 4, 31),
                         (2, 5, 37),
                         (2, 6, 15),
                         (3, 4, 19),
                         (3, 5, 42),
                         (3, 6, 23),
                         (4, 5, 24),
                         (4, 6, 17),
                         (5, 6, 24)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(2, 4),
                 (3, 4),
                 (4, 6),
                 (5, 6)]
            P = [63, 49, 45, 53]
        if var_2==6:
            distances = [(1, 2, 22),
                         (1, 3, 43),
                         (1, 4, 39),
                         (1, 5, 28),
                         (1, 6, 20),
                         (2, 3, 26),
                         (2, 4, 33),
                         (2, 5, 36),
                         (2, 6, 17),
                         (3, 4, 22),
                         (3, 5, 40),
                         (3, 6, 24),
                         (4, 5, 22),
                         (4, 6, 19),
                         (5, 6, 20)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(2, 4),
                 (4, 6),
                 (3, 5),
                 (5, 2)]
            P = [51, 23, 29, 31]
        if var_2==7:
            distances = [(1, 2, 24),
                         (1, 3, 41),
                         (1, 4, 36),
                         (1, 5, 22),
                         (1, 6, 19),
                         (2, 3, 21),
                         (2, 4, 33),
                         (2, 5, 33),
                         (2, 6, 14),
                         (3, 4, 27),
                         (3, 5, 39),
                         (3, 6, 23),
                         (4, 5, 20),
                         (4, 6, 20),
                         (5, 6, 19)]

            V = [1, 3, 4, 5, 6, 2, 1]
            Z = [(3, 4),
                 (4, 6),
                 (5, 2),
                 (6, 2)]
            P = [33, 82, 51, 76]
        if var_2==8:
            distances = [(1, 2, 19),
                         (1, 3, 39),
                         (1, 4, 35),
                         (1, 5, 26),
                         (1, 6, 18),
                         (2, 3, 26),
                         (2, 4, 33),
                         (2, 5, 37),
                         (2, 6, 14),
                         (3, 4, 22),
                         (3, 5, 41),
                         (3, 6, 21),
                         (4, 5, 22),
                         (4, 6, 19),
                         (5, 6, 24)]

            V = [1, 4, 2, 3, 5, 6, 1]
            Z = [(5, 1),
                 (4, 5),
                 (2, 3),
                 (3, 4)]
            P = [88, 54, 24, 64]
        if var_2==0:
            continue
        T = 100

        def probability(delta, T):
            return 100 * math.e ** (-delta / T)

        def reductTemp(prevT):
            nextT = 0.5 * prevT
            return nextT

        def edgeLength(i, j, distances, roundTrip=True):
            if roundTrip:
                return max([(item[2] if (item[0] == i and item[1] == j) or (item[1] == i and item[0] == j) else -1)
                            for item in distances])
            else:
                return max([(item[2] if (item[0] == i and item[1] == j) else -1) for item in distances])

        def routeLength(V, distances):
            edges = []
            for i in range(len(V) - 1):
                edges.append(edgeLength(V[i], V[i + 1], distances))
            return sum(edges)

        def routeOneReplacement(arrV, Z, replacementByName=True):
            decrement = 1 if replacementByName else 0
            arrV[Z[0] - decrement], arrV[Z[1] - decrement] = arrV[Z[1] - decrement], arrV[Z[0] - decrement]
            return arrV

        def routeReplacement(V, Z):
            for z in Z:
                V = routeOneReplacement(V, z)
            return V

        def chooseRoute(distances, V, Z, T, P):
            sumLength = routeLength(V, distances)
            arrSum = [sumLength]
            for i in range(len(Z)):
                newV = routeOneReplacement(V[:], Z[i])
                #newS = routeLength(newV, distances)
                newS=random.randint(1, 1000)
                print(f"random newS = {newS}")
                arrSum.append(newS)
                deltaS = newS - sumLength
                if deltaS > 0:
                    p = probability(deltaS, T)
                    if p > P[i]:
                        V = newV
                        sumLength = newS
                else:
                    V = newV
                    sumLength = newS
                T = reductTemp(T)
            return V, arrSum

        def drawRouteGraph(distances, bestRoute):
            newDistances = []
            for i in range(len(bestRoute) - 1):
                for distance in distances:
                    if distance[0] == bestRoute[i] and distance[1] == bestRoute[i + 1] or distance[1] == bestRoute[i] and \
                            distance[0] == bestRoute[i + 1]:
                        newDistances.append(distance)
            graph = nx.Graph()
            graph.add_weighted_edges_from(newDistances)
            nx.draw_kamada_kawai(graph, node_color='#fb7258', node_size=2000, with_labels=True)
            print("done")

        bestRoute, arrLength = chooseRoute(distances, V, Z, T, P)

        print(f'Лучший выбранный маршрут: {bestRoute}')
        print(f'Длина лучшего выбранного маршрута: {routeLength(bestRoute, distances)}')
        print(f'Длины всех рассмотренных маршрутов: {arrLength}')

def main():
    global num_1_cycle
    print(task_list)
    current_task=str(input("Введите номер задания: "))
    if current_task=="1":
        task_1()
        num_1_cycle=1
    if current_task=="2":
        task_2()
    if current_task=="3":
        task_3()
    if current_task=="0":
        exit()

while True:
    main()