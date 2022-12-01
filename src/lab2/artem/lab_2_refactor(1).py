#!/usr/bin/env python
# coding: utf-8

# ## Задача 1.2. Точки (Р-модель распознавания)
#
# ### Постановка задачи
# Пусть образы объектов описываются группами из двух целочисленных
# параметров $(x, y)$. Имеется два непересекающихся класса объектов. Требуется провести границу между классами. Способ построения разграничивающей прямой предлагается разработать самостоятельно.
#
# ### Исходные данные
# Два натуральных числа $N_1$ – количество образцов из первого класса и $N_2$ – количество образцов из второго класса. $N1 + N2$ пар чисел $(x_k, y_k)$ для образцов из первого и второго классов.
#
# Требуется выполнить графическую иллюстрацию Р-модели
#
# > Замечание.
# >
# > Точки разных классов могут задаваться пользователем произвольно
# или генерироваться автоматически.
# Для автоматического формирования наборов точек $(x_k, y_k)$ каждого класса следует воспользоваться следующей информацией. Пусть в пространстве признаков $R^2$ заданы два нормальных распределения с математическими  жиданиями $(Mx_1, My_1)$ и $(Mx_2, My_2)$ и дисперсиями $σ_1$ и $σ_2$.
# >
# > Каждое из распределений задает один из классов объектов. Производится случайный выбор точек (объектов) и разыгрывается по заданным законам класс, в который они зачисляются. После того, как определены $N_1 + N_2$
# объектов, считаем, что исходная информация задана.
# >
# > Таким образом, при разработке программы следует предусмотреть ввод
# пользователем величин $N_1$, $N_2$, $Mx_1$, $My_1$, $Mx_2$, $My_2$, $σ_1$ и $σ_2$.
#
# Работу выполнил: студент группы ПИИ(м)-21, Латынцев А.В.
#
# http://mathprofi.ru/uravnenie_pryamoi_na_ploskosti.html
#
#

# ![round.resized.jpeg](attachment:round.resized.jpeg)

# ![slide-1.resized.jpg](attachment:slide-1.resized.jpg)

# In[16]:


# Подключаем необходимые пакеты
import scipy.stats as sps  # 1.9.2
import numpy as np  # 1.23.4
import ipywidgets as widgets  # 8.0.2
import matplotlib.pyplot as plt  # 3.6.1
import matplotlib.lines as mlines
from sklearn import svm  # 1.1.2
from sklearn.linear_model import LogisticRegression
from math import cos, pi, sin, asin, acos, atan, tan, sqrt, degrees
# использовать системное приложение для взаимодействия с графиками
# в моем случае - TkAgg
# get_ipython().run_line_magic('matplotlib', '')


def grad_to_rad(grad):
    return grad / 360 * pi * 2


# In[17]:


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

    def poly_f(self):
        fit = np.polyfit(self.points[:, 0], self.points[:, 1], 3)
        return fit

    # Метод для ручного добавления точек
    def add_point(self, x, y):
        self.points = list(self.points)
        self.points.append(np.array([x, y]))
        self.points = np.array(self.points)

    # Вернуть значение функции плотности вероятности для точек
    # (по факту считается очень маленький интервал)
    def return_probability(self, x, y):
        return self.norm_distribution.pdf(np.array([x, y]))


# In[18]:


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
        self.add_hyperplane(line_color='#001B33')
        if show:
            self.ax.plot()

        plt.show()  # показать график

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

    def add_hyperplane(self, line_color="#00A65D"):
        x_1, y_1 = self.object_1.math_exp
        x_2, y_2 = self.object_2.math_exp
        mid_x = x_2  # (x_1 + x_2)/2
        mid_y = y_2  # (y_1 + y_2)/2
        self.ax.scatter([[mid_x]], [mid_y], color='#FDB94D')

        # ang = degrees(acos( (mid_x - x_1) / sqrt((mid_x - x_1)**2 + (mid_y - y_1)**2) ))
        ang = degrees(atan((mid_y - y_1) / (mid_x - x_1)))
        normal_ang = 90 + ang

        def x(grad): return cos(grad_to_rad(grad)) + mid_x

        def y(grad): return sin(grad_to_rad(grad)) + mid_y
        normal_x = x(normal_ang)
        normal_y = y(normal_ang)

        # y = kx + b                                                # p_N = x_n, y_n;
        def k(p_1, p_2): return (p_2[1] - p_1[1]) / (p_2[0] - p_1[0])  # p_2 = x_2, y_2;

        def b(p_1, p_2): return (p_1[1] * p_2[0] - p_1[0] * p_2[1]) / (p_2[0] - p_1[0])
        k_normal = k((mid_x, mid_y), (normal_x, normal_y))
        b_normal = b((mid_x, mid_y), (normal_x, normal_y))
        # print(f"normal_ang = { normal_ang } или { 180 + degrees(atan(k_normal)) }")
        # print(f"y = {round(k_normal, 3)} * x + {b_normal}")

        x_points = list(map(lambda value: value[0], list(self.object_1.points) + list(self.object_2.points)))
        x_min, x_max = round(np.min(x_points), 2), round(np.max(x_points), 2)

        new_y_norm_left_point = k_normal * x_min + b_normal
        new_y_norm_right_point = k_normal * x_max + b_normal

        self.ax.add_line(
            mlines.Line2D(
                [x_min, x_max],
                [new_y_norm_left_point, new_y_norm_right_point],
                color=line_color,
                marker="x")
        )
        # print(f"[{x_min, x_max}], [{new_y_norm_left_point, new_y_norm_right_point}]")

        # ------------------
        k_base = k((x_1, y_1), (x_2, y_2))
        b_base = b((x_1, y_1), (x_2, y_2))
        print(f"k_normal = {k_normal}, b_normal = {b_normal}")
        print(f"k_base = {k_base}, b_base = {b_base}")

        # радостно шагаем по линии, соединяющей центры
        total_steps = 10
        increment = (x_2 - x_1) / 10
        g_counter = []
        for step in range(total_steps + 1):
            offset_x_1_normal_line = mid_x - step * increment
            offset_y_1_normal_line = k_base * offset_x_1_normal_line + b_base
            self.ax.scatter([[offset_x_1_normal_line]], [offset_y_1_normal_line], color='#FDB94D')

            # np.array(list(map(lambda value: value[0], self.object_1.points))),
            # np.array(list(map(lambda value: value[1], self.object_1.points)))
            counter_1 = 0
            r_ang_1 = []
            for point in self.object_1.points:
                oy_offset = point[1] - offset_y_1_normal_line
                relateve_ang = degrees(atan((oy_offset) / (offset_x_1_normal_line)))
                r_ang_1.append(round(relateve_ang, 0))
                # print(round(relateve_ang, 0))
                if relateve_ang < 0:
                    counter_1 += 1

            counter_2 = 0
            r_ang_2 = []
            for point in self.object_2.points:
                oy_offset = point[1] - offset_y_1_normal_line
                relateve_ang = degrees(atan((offset_x_1_normal_line) / (oy_offset)))
                r_ang_2.append(round(relateve_ang, 0))
                # print(round(relateve_ang, 0))
                if relateve_ang > 0:
                    counter_2 += 1
            print(f"r_ang_1 = {r_ang_1}")
            print(f"r_ang_2 = {r_ang_2}")
            g_counter.append([counter_1, counter_2])

        for step in range(len(g_counter)):
            print(f"{step}: {g_counter[step]}, sum = {g_counter[step][0] + g_counter[step][1]}")


# In[19]:


if __name__ == "__main__":

    # Создаем два "облака"
    cloud_1 = norm_distribution(
        N=100,
        math_exp=[1, 1],
        #cov_m=[[1, 0], [0, 1]]
        cov_m=[[10, 0], [0, 1000]]
    )

    cloud_2 = norm_distribution(
        N=100,
        math_exp=[30, 40],
        cov_m=[[20, 0], [0, 20]]
    )

    # cloud_1.show_3d_plot_of_distribution()


# In[20]:


# Инициализация сравнения двух "облаков"
cloud_comparison = distribution_analysis(
    cloud_1, cloud_2, graph_title='Облака образов'
)
# Построение и отображение графика
cloud_comparison.plot_everything(show=True)
