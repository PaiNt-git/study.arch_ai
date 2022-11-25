# %%
"""
## Задача 1.2. Точки (Р-модель распознавания)

### Постановка задачи
Пусть образы объектов описываются группами из двух целочисленных
параметров $(x, y)$. Имеется два непересекающихся класса объектов. Требуется провести границу между классами. Способ построения разграничивающей прямой предлагается разработать самостоятельно.

### Исходные данные
Два натуральных числа $N_1$ – количество образцов из первого класса и $N_2$ – количество образцов из второго класса. $N1 + N2$ пар чисел $(x_k, y_k)$ для образцов из первого и второго классов.

Требуется выполнить графическую иллюстрацию Р-модели

> Замечание.
>
> Точки разных классов могут задаваться пользователем произвольно
или генерироваться автоматически.
Для автоматического формирования наборов точек $(x_k, y_k)$ каждого класса следует воспользоваться следующей информацией. Пусть в пространстве признаков $R^2$ заданы два нормальных распределения с математическими  жиданиями $(Mx_1, My_1)$ и $(Mx_2, My_2)$ и дисперсиями $σ_1$ и $σ_2$.
>
> Каждое из распределений задает один из классов объектов. Производится случайный выбор точек (объектов) и разыгрывается по заданным законам класс, в который они зачисляются. После того, как определены $N_1 + N_2$
объектов, считаем, что исходная информация задана.
>
> Таким образом, при разработке программы следует предусмотреть ввод
пользователем величин $N_1$, $N_2$, $Mx_1$, $My_1$, $Mx_2$, $My_2$, $σ_1$ и $σ_2$.

Работу выполнил: студент группы ПИИ(м)-21, Латынцев А.В.
"""

# %%
# Подключаем необходимые пакеты
import scipy.stats as sps  # 1.9.2
import numpy as np  # 1.23.4
import ipywidgets as widgets  # 8.0.2
import matplotlib.pyplot as plt  # 3.6.1
import matplotlib.lines as mlines
from sklearn import svm  # 1.1.2
from sklearn.linear_model import LogisticRegression
from math import cos, pi, sin
# использовать системное приложение для взаимодействия с графиками
# в моем случае - TkAgg
# %matplotlib


def grad_to_rad(grad):
    return grad / 360 * pi * 2

# %%
# класс для создания объектов 2-мерного нормального распределения


class norm_distribution:
    def __init__(self, N: int, math_exp: list, cov_m: list):
        """
        N: кол-во образцов из первого класса
        math_exp: математическое ожидание [Mx1, My1]
        cov: ковариацион. матрица распределения
             [[variance_x, 0], [0, variance_y]]
        """

        self.math_exp = math_exp
        self.cov_m = cov_m

        # Объект нормального распределения
        self.norm_distribution = \
            sps.multivariate_normal(mean=math_exp, cov=cov_m)

        self.points = \
            self.norm_distribution.rvs(size=N)  # наши N точки

    # Метод для ручного добавления точек
    def add_point(self, x, y):
        self.points = list(self.points)
        self.points.append(np.array([x, y]))
        self.points = np.array(self.points)

    # Вернуть значение функции плотности вероятности для точек
    # (по факту считается очень маленький интервал)
    def return_probability(self, x, y):
        return self.norm_distribution.pdf(np.array([x, y]))

    # Просто для удобства
    def show_points(self, count_of_points=0):
        if count_of_points == 0:
            print("Значения выборки:\n", self.points[:])
        else:
            print(
                f"Первые {count_of_points} значений выборки:\n",
                self.points[:count_of_points]
            )

    # Просто для удобства
    def show_3d_plot_of_distribution(self, x_domain=0, y_domain=0):
        """
        x_domain и y_domain
            конечные координаты на 3D графике соответственно;
            по умолчанию - 0, что означает - подобрать самостоятельно
        """

        # Предустановка масштаба, если нужно
        if x_domain == 0:
            self.x_domain = (self.cov_m[0][0] + self.math_exp[0]) * 2
        else:
            self.x_domain = x_domain
        if y_domain == 0:
            self.y_domain = (self.cov_m[1][1] + self.math_exp[1]) * 2
        else:
            self.y_domain = y_domain

        # Создаем сетку и многомерную нормаль
        x, y = \
            np.linspace(-self.x_domain, self.x_domain, 500), \
            np.linspace(-self.y_domain, self.y_domain, 500)
        X, Y = np.meshgrid(x, y)  # компануем в сетку

        # Определяем ось вероятностей
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        Z = self.norm_distribution.pdf(pos)

        # Строем 3D график
        fig = plt.figure(figsize=(10, 10))
        self.ax = fig.add_subplot(projection='3d')
        self.ax.plot_surface(
            X, Y, Z,
            cmap='viridis', linewidth=0
        )

        # Установка осей
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        plt.show()  # показать график

# %%


class distribution_analysis:
    def __init__(self, object_1, object_2, graph_title: str):
        """
        object_1 и object_2 - экземпляры класса norm_distribution
        """
        self.object_1, self.object_2 = object_1, object_2

        # Построение Координатной плоскости облака образов
        fig, self.ax = plt.subplots(
            figsize=(10, 10),
            num=graph_title
        )
        self.initialization_of_graph()

    def plot_everything(self, show: bool):
        self.add_norm_points_on_graph()
        self.add_connect_centers_line()
        self.add_hyperplane(self.get_points_of_hyperplane(obj=self.object_1))
        self.add_hyperplane(self.get_points_of_hyperplane(obj=self.object_2), line_color='#001B33')

        if show:
            self.ax.plot()

        plt.show()

    def initialization_of_graph(self):
        self.ax.set_aspect('equal', adjustable='box')

        # Удаление верхней и правой границ
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Добавление основных линий сетки
        self.ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    def add_norm_points_on_graph(self, points_color_1='#454FA1', points_color_2='#E01A2D'):
        self.ax.scatter(
            np.array(list(map(lambda value: value[0], self.object_2.points))),
            np.array(list(map(lambda value: value[1], self.object_2.points))),
            color=points_color_2
        )  # x2, y2

        self.ax.scatter(
            np.array(list(map(lambda value: value[0], self.object_1.points))),
            np.array(list(map(lambda value: value[1], self.object_1.points))),
            color=points_color_1
        )

    def add_connect_centers_line(self):
        # для удобства
        x1_m, y1_m = self.object_1.math_exp[0], self.object_1.math_exp[1]
        x2_m, y2_m = self.object_2.math_exp[0], self.object_2.math_exp[1]

        lM = mlines.Line2D(
            [x1_m, x2_m], [y1_m, y2_m],
            color="#000", linestyle="--", marker="x"
        )
        self.ax.add_line(lM)

        self.ax.annotate(f'({x1_m}; {y1_m})',
                         (x1_m, y1_m),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         color='blue', backgroundcolor="#eae1e196")
        self.ax.annotate(f'({x2_m}; {y2_m})',
                         (x2_m, y2_m),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         color='blue', backgroundcolor="#eae1e196")

    def get_points_of_hyperplane(self, obj, margin_of_error=0.0005):
        x_m, y_m = obj.math_exp[0], obj.math_exp[1]
        if x_m > y_m:
            step_accuracy = x_m
        else:
            step_accuracy = y_m

        # без этого критерий остановки зависит от параметров матрицы ковариации
        # ("чем шире дисперсия", тем меньше вероятность быть у мат. ожидания)
        # (наверное, глупость сморозил?!)
        if obj.cov_m[0][0] > obj.cov_m[1][1]:
            margin_of_error = margin_of_error / obj.cov_m[0][0]
        else:
            margin_of_error = margin_of_error / obj.cov_m[1][1]

        # насколько точно определим крайние 2 точки
        step_x = abs(x_m / step_accuracy)
        step_y = abs(y_m / step_accuracy)

        # определяем наибольшие точки сверху
        ITER = 0
        current_y = y_m + step_y * ITER
        while abs(obj.return_probability(x_m, current_y)) > margin_of_error:
            current_y = y_m + step_y * ITER
            ITER += 1
        top_y = step_y * ITER  # + obj.cov_m[1][1] # current_y

        # определяем наибольшие точки справа
        ITER = 0
        current_x = x_m + step_x * ITER
        while abs(obj.return_probability(current_x, y_m)) > margin_of_error:
            current_x = x_m + step_x * ITER
            ITER += 1
        top_x = step_x * ITER  # + obj.cov_m[0][0]*0.50 # current_x  obj.cov_m[0][0]

        xy_points = []
        pois = []
        for i in range(0, 361, 30):
            xy_points.append([x_m + top_x * cos(grad_to_rad(i)), y_m + top_y * sin(grad_to_rad(i))])

        return xy_points

    def add_hyperplane(self, points, line_color="#00A65D"):
        """
        points - результат работы метода get_points_of_hyperplane
        """
        x_points, y_points = \
            np.array(list(map(lambda value: value[0], points))), \
            np.array(list(map(lambda value: value[1], points)))

        # self.ax.scatter(x_points, y_points, color=line_color)

        for i in range(1, len(points), 1):
            self.ax.add_line(
                mlines.Line2D(
                    [x_points[i - 1], x_points[i]],
                    [y_points[i - 1], y_points[i]],
                    color=line_color,
                    marker="x")
            )



# %%
if __name__ == "__main__":

    # Создаем два "облака"
    cloud_1 = norm_distribution(
        N=5000,
        math_exp=[1, 100],
        #cov_m=[[1, 0], [0, 1]]
        cov_m=[[1, 0], [0, 200]]
    )

    cloud_2 = norm_distribution(
        N=500,
        math_exp=[100, 150],
        cov_m=[[10, 0], [0, 10]]
    )

    # cloud_1.show_3d_plot_of_distribution()


# %%
    # Инициализация сравнения двух "облаков"
    cloud_comparison = distribution_analysis(
        cloud_1, cloud_2, graph_title='Облака образов'
    )
    # Построение и отображение графика
    cloud_comparison.plot_everything(show=True)

# %%
