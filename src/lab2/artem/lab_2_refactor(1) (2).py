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

# In[636]:


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
get_ipython().run_line_magic('matplotlib', '')

def grad_to_rad(grad):
    return grad/360*pi*2


# In[637]:


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
            self.norm_distribution.rvs(size=N) # наши N точки
    
    def poly_f(self):
        fit = np.polyfit(self.points[:,0], self.points[:,1], 3)
        return fit
    
    # Метод для ручного добавления точек
    def add_point(self, x, y):
        self.points = list(self.points)
        self.points.append(np.array([x, y]))
        self.points = np.array(self.points)

    # Вернуть значение функции плотности вероятности для точек
    # (по факту считается очень маленький интервал)
    def return_probability(self, x, y):
        return self.norm_distribution.pdf(np.array([x,y]))


# In[638]:


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

    
    def return_min_and_max_from_2_set(self, for_x=True):
        if for_x:
            index = 0
        else:
            index = 1
            
        merge_points = list(
            map(
                lambda value: value[index], 
                list(self.object_1.points) + list(self.object_2.points)
            )
        )
        return round(np.min(merge_points), 2), round(np.max(merge_points), 2)
   
    
    def add_hyperplane(self, line_color="#00A65D"):
        # задаем центральную линию и определяем набор точек на ней с определенным шагом
        line_connecting_centers = straight_line(self.object_1.math_exp, self.object_2.math_exp, color="#FDB94D")
        line_connecting_centers.get_intermediate_points(show_in = None, need_points=20)  # show_in = self.ax
        
        # --- цикл нахождения наилучшей точки для построения перпендикуляра ---
        best_point = line_connecting_centers.intermediate_points[0]  # лучшая точка - первая
        previously_count_of_good_point = 0  # количество хороших точек на предыдущем шаге
        
        # чтобы точно знать слева или справа находятся хорошие точки
        # (шагаем-то всегда от мат. ожидания второго распределения к первому)
        if self.object_2.math_exp[0] > self.object_1.math_exp[0]:
            p_1 = self.object_2.points
            p_2 = self.object_1.points
        else:
            p_1 = self.object_1.points
            p_2 = self.object_2.points
        
        for point in line_connecting_centers.intermediate_points:

            count_of_good_point = 0  # количество хороших точек на этом шаге
            
            for point_set_2 in p_1:
                ang = straight_line.return_right_atan_ang(point, point_set_2)
                # если справа
                if 90 + line_connecting_centers.atan_ang > ang or ang > 270 + line_connecting_centers.atan_ang:
                    count_of_good_point += 1
                    if False:  # debug
                        self.ax.scatter(
                            [ point_set_2[0] ], 
                            [ point_set_2[1] ], 
                            color='#FDB94D'
                        )
                        
            for point_set_2 in p_2:
                ang = straight_line.return_right_atan_ang(point, point_set_2)
                # если слева
                if 90 + line_connecting_centers.atan_ang < ang < 270 + line_connecting_centers.atan_ang:
                    count_of_good_point += 1
                    
            if previously_count_of_good_point < count_of_good_point:
                previously_count_of_good_point = count_of_good_point
                best_point = point
        # ----------------------------------------------------------
            
        # задаем уравнение перпендикулярной line_connecting_centers прямой для точки best_point
        starting_line_on_LCC = straight_line(
            best_point, 
            straight_line.return_rotated_point(
                line_connecting_centers.atan_ang + 90,
                best_point
            ),
            color="#00A65D"
        )
        
        # определяемся с границами и строим перпендикуляр через best_point
        x_range = abs(abs(self.object_1.math_exp[0]) - abs(self.object_2.math_exp[0]))
        y_range = abs(abs(self.object_1.math_exp[1]) - abs(self.object_2.math_exp[1]))
        if x_range < y_range:
            self.x_min_max = self.return_min_and_max_from_2_set()
            starting_line_on_LCC.draw_line_on_the_chart(self.ax, x_min_max=self.x_min_max)
        else:
            self.y_min_max = self.return_min_and_max_from_2_set(for_x=False)
            starting_line_on_LCC.draw_line_on_the_chart(self.ax, y_min_max=self.y_min_max)



# In[639]:


class straight_line:
    @staticmethod
    def return_rotated_point(ang, offset):
        """offset: [x, y]"""
        x = lambda grad: cos(grad_to_rad(grad))  + offset[0]
        y = lambda grad: sin(grad_to_rad(grad))  + offset[1]
        return x(ang), y(ang)

    @staticmethod
    def return_atan_ang(point_1, point_2):
        """
        1 и 3 четверть - от 0 до 90
        2 и 4 четверть - от -90 до 0
        """
        return degrees(atan(  (point_2[1] - point_1[1]) / (point_2[0] - point_1[0]) ))
    
    @staticmethod
    def return_right_atan_ang(start_point, end_point):
        """
        1 четверть - от 0 до 90
        2 четверть - от 90 до 180
        3 четверть - от 180 до 270
        4 четверть - от 270 до 360
        """
        degrees = straight_line.return_atan_ang(start_point, end_point)
        if end_point[0] > start_point[0] and end_point[1] > start_point[1]:  # 1
            return degrees
        elif end_point[0] < start_point[0] and end_point[1] > start_point[1]:  # 2
            return 180 + degrees
        elif end_point[0] < start_point[0] and end_point[1] < start_point[1]:  # 3
            return 180 + degrees 
        else:  # 4
            return 360 + degrees
    
    def __init__(self, point_1, point_2, color="#454FA1"):
        self.k_func = lambda p_1, p_2: (p_2[1] - p_1[1]) / (p_2[0] - p_1[0])  # p_2 = x_2, y_2;
        self.b_func = lambda p_1, p_2: (p_1[1] * p_2[0] - p_1[0] * p_2[1] ) / (p_2[0] - p_1[0])
        
        self.point_1 = point_1
        self.point_2 = point_2
        self.k = self.k_func( self.point_1, self.point_2 )
        self.b = self.b_func( self.point_1, self.point_2 )
        self.atan_ang = self.return_atan_ang( self.point_1, self.point_2 )
        self.color = color
        
        self.intermediate_points = []

 
    def get_intermediate_points(self, show_in=None, need_points=10):
        """
        Данный метод формирует массив точек по всей длине 
        исходной линии, начиная и заканчивая крайними 
        self.point_1/self.point_2. Записывает в список
        self.intermediate_points
        """
        self.intermediate_points = []
        total_steps = need_points
        increment_x = (self.point_2[0] - self.point_1[0]) / total_steps
        increment_y = (self.point_2[1] - self.point_1[1]) / total_steps
        
        for step in range(total_steps + 1):
            new_x = self.point_2[0] - step * increment_x
            new_y = self.point_2[1] - step * increment_y  # self.k * new_x + self.b
            self.intermediate_points.append([new_x, new_y])
            
        if show_in is not None:
            self.ax = show_in
            self.ax.scatter(
                list(map(lambda x: x[0], self.intermediate_points)), 
                list(map(lambda y: y[1], self.intermediate_points)), 
                color=self.color  # '#FDB94D'
            )
   
    def get_y_from_x(self, x):
        return self.k * x + self.b
    
    def get_x_from_y(self, y):
        return (y - self.b) / self.k

    def draw_line_on_the_chart(self, plt_obj, x_min_max=None, y_min_max=None):
        """
        x_min_max: [x_min, x_max]
           если указано, 
              сначала вычислить точки прямой, соответствующие данным координатам,
              после - построить линию
           иначе построить по точкам внутри экземпляра класса
        
        """
        if x_min_max is not None:
            point_1 = ( x_min_max[0], self.get_y_from_x(x_min_max[0]) )
            point_2 = ( x_min_max[1], self.get_y_from_x(x_min_max[1]) )
        elif y_min_max is not None:
            point_1 = ( self.get_x_from_y(y_min_max[0]), y_min_max[0] )
            point_2 = ( self.get_x_from_y(y_min_max[1]), y_min_max[1] )      
        else:
            point_1 = self.point_1
            point_2 = self.point_2
            
            
        self.ax = plt_obj
        self.ax.scatter(
            [ point_1[0], point_2[0] ], 
            [ point_1[1], point_2[1] ], 
            color=self.color  # '#FDB94D'
        )
        
        self.ax.add_line(
            mlines.Line2D(
                [ point_1[0], point_2[0] ], 
                [ point_1[1], point_2[1] ], 
                color=self.color, 
                marker="x")
        )
        
    def print_the_equation_of_a_straight_line(self):
        print(f"y = {round(self.k, 1)} x + {round(self.b, 1)}")


# In[640]:


if __name__ == "__main__":
    
    # Создаем два "облака"
    cloud_1 = norm_distribution(
        N=50, 
        math_exp=[1, 3], 
        #cov_m=[[1, 0], [0, 1]]
        cov_m=[[50, 0], [0, 10]]
    )
    
    cloud_2 = norm_distribution(
        N=50, 
        math_exp=[20, 5],
        cov_m=[[20, 0], [0, 20]]
    )

    
    # cloud_1.show_3d_plot_of_distribution()
    


# In[641]:


# Инициализация сравнения двух "облаков"
cloud_comparison = distribution_analysis(
    cloud_1, cloud_2, graph_title='Облака образов'
)
# Построение и отображение графика
cloud_comparison.plot_everything(show=True)

