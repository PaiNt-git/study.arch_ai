#!/usr/bin/env python
# coding: utf-8

# # Полезные ссылки по теме:
#
# [Многомерное нормальное распределение](https://ru.wikipedia.org/wiki/Многомерное_нормальное_распределение)
#
# [Создание и построение 3d гауссовых распределений](https://machinelearningmastery.ru/a-python-tutorial-on-generating-and-plotting-a-3d-guassian-distribution-8c6ec6c41d03/)
#
# [Plotting a decision boundary separating 2 classes using Matplotlib's pyplot](https://stackoverflow.com/questions/22294241/plotting-a-decision-boundary-separating-2-classes-using-matplotlibs-pyplot)
#
# ![3.png](attachment:3.png)

# Подходы к разделению бывают [разными](https://scikit-learn.org/0.18/modules/ensemble.html)...
# ![4.png](attachment:4.png)

# In[9]:


import scipy.stats as sps  # 1.9.2
import numpy as np  # 1.23.4
import ipywidgets as widgets  # 8.0.2
import matplotlib.pyplot as plt  # 3.6.1
import matplotlib.lines as mlines
from sklearn import svm
from sklearn.linear_model import LogisticRegression
#%matplotlib inline
# Графики в GUI (TkAgg)
# get_ipython().run_line_magic('matplotlib', '')


# In[10]:


def build_graph_with_points(object_1, object_2, with_clf=False):

    x1_points, y1_points, x2_points, y2_points = \
        np.array(list(map(lambda value: value[0], object_1.points))), \
        np.array(list(map(lambda value: value[1], object_1.points))), \
        np.array(list(map(lambda value: value[0], object_2.points))), \
        np.array(list(map(lambda value: value[1], object_2.points)))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 10), num='Облака образов')
    ax.set_aspect('equal', adjustable='box')

    # Удаление верхней и правой границ
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Добавление основных линий сетки
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # Образы
    ax.scatter(x2_points, y2_points, color="#454FA1")  # blue
    ax.scatter(x1_points, y1_points, color="#E01A2D")  # red

    # Моя гиперплоскость
    xy_points = calculate_new_points_of_the_dividing_line(object_1, object_2)
    x3_points, y3_points = \
        np.array(list(map(lambda value: value[0], xy_points))), \
        np.array(list(map(lambda value: value[1], xy_points)))
    #ax.scatter(x3_points, y3_points, color="#00A65D")  # green
    for i in range(1, len(xy_points), 1):
        ax.add_line(
            mlines.Line2D(
                [x3_points[i - 1], x3_points[i]],
                [y3_points[i - 1], y3_points[i]],
                color="#00A65D",
                marker="x")
        )

    # линия, соединяющая центры "облаков"
    x1m, x2m, y1m, y2m = \
        object_1.math_exp[0], \
        object_2.math_exp[0], \
        object_1.math_exp[1], \
        object_2.math_exp[1]
    lM = mlines.Line2D([x1m, x2m], [y1m, y2m], color="#000", linestyle="--", marker="x")
    ax.add_line(lM)
    ax.annotate(f'({x1m}; {y1m})',
                (x1m, y1m),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")
    ax.annotate(f'({x2m}; {y2m})',
                (x2m, y2m),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")

    #ax.plot()

    # ----------------- from sklearn import svm # -----------------
    if with_clf:
        X = np.concatenate((object_1.points, object_2.points), axis=0)
        Y = np.array([0] * len(object_1.points) + [1] * len(object_2.points))

        mode_is = 1

        if mode_is == 1:  # линейный случай
            C = 1.0  # SVM regularization parameter
            clf = svm.SVC(kernel='linear', gamma=0.7, C=C)
            # clf = LogisticRegression()  # метод логич. регрессии
            clf.fit(X, Y)

            # Коэффициент признаков в решающей функции. (от тета 1 до тета n)
            w = clf.coef_[0]

            # Перехват (также известный как смещение)
            # добавлен в функцию принятия решения. (тета 0)
            w0 = clf.intercept_

            # координаты линии по осям
            xx = [np.min(X[:, 1] - 5), np.max(X[:, 1] + 5)]
            yy = np.dot((-1. / w[1]), (np.dot(w[0], xx) + w0))

            ax.plot(xx, yy, 'k-')

        elif mode_is == 2:
            C = 1.0  # SVM regularization parameter
            clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
            clf.fit(X, Y)

            h = .02  # step size in the mesh
            # create a mesh to plot in
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contour(xx, yy, Z, cmap=plt.cm.Paired)
            # ax.plot(xx, yy, 'k-')

    plt.show()


# In[11]:


class object_image:
    def __init__(self):
        self.points = list()
        self.norm_distribution = None

    def create_norm_distribution(self, N: int, math_exp: list, cov_m: list):
        """
        N: кол-во образцов из первого класса
        mean: математическое ожидание [Mx1, My1]
        cov: ковариацион. матрица распределения
             [[variance_x, 0], [0, variance_y]]
             диапазон(domain)*2/variance(дисперсия(ширина))
        """

        self.math_exp = math_exp
        self.cov_m = cov_m

        self.norm_distribution = \
            sps.multivariate_normal(mean=math_exp, cov=cov_m)  # шаблон

        self.points = \
            self.norm_distribution.rvs(size=N)  # наши N точки

    def show_points(self, count_of_points=0):
        if count_of_points == 0:
            print("Значения выборки:\n", self.points[:])
        else:
            print(
                f"Первые {count_of_points} значений выборки:\n",
                self.points[:count_of_points]
            )

    def show_3d_plot_of_distribution(self, x_domain=0, y_domain=0):
        # предустановка масштаба, если нужно
        if x_domain == 0:
            self.x_domain = (self.cov_m[0][0] + self.math_exp[0]) * 2
        else:
            self.x_domain = x_domain
        if y_domain == 0:
            self.y_domain = (self.cov_m[1][1] + self.math_exp[1]) * 2
        else:
            self.y_domain = y_domain

        #Create grid and multivariate normal
        x = np.linspace(-self.x_domain, self.x_domain, 500)  # в диапазоне [-10; 10] 500 точек
        y = np.linspace(-self.y_domain, self.y_domain, 500)
        X, Y = np.meshgrid(x, y)  # скомпоновать в сетку

        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X; pos[:, :, 1] = Y
        Z = self.norm_distribution.pdf(pos)

        #R = np.sqrt(X**2 + Y**2)
        #Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))

        #Make a 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap='viridis', linewidth=0
        )

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()

    def add_point(self, x, y):
        self.points = list(self.points)
        self.points.append(np.array([x, y]))
        self.points = np.array(self.points)

    def return_probability(self, x, y):
        return self.norm_distribution.pdf(np.array([x, y]))


# In[12]:


def calculate_new_points_of_the_dividing_line(object_1, object_2):

    count_of_steps = 30

    # для удобства
    x_1, y_1 = object_1.math_exp[0], object_1.math_exp[1]
    x_2, y_2 = object_2.math_exp[0], object_2.math_exp[1]

    delta_x = x_1 - x_2
    delta_y = y_1 - y_2
    step_x, step_y = -delta_x / count_of_steps, -delta_y / count_of_steps

    margin_of_error = 0.000005
    return_points = []

    if delta_x == 0 and delta_y != 0:
        pass
        # идем только по y, берем перпендикуляр
    elif delta_y == 0 and delta_x != 0:
        pass
        # идем только по y, берем перпендикуляр
    elif delta_y != 0 and delta_x != 0:
        # 2d случай
        # print(y_1, y_2, step_y)
        # print(x_1, x_2, step_x)
        for step_1 in range(count_of_steps):
            for step_2 in range(count_of_steps):
                current_x = step_x * step_2 + x_1
                current_y = step_y * step_1 + y_1
                if abs(object_1.return_probability(current_x, current_y) - object_2.return_probability(current_x, current_y)) <= margin_of_error:
                    return_points.append([current_x, current_y])
                    break

        part_2 = []
        #print('return_points =', return_points)
        for x, y in return_points:
            part_2.append([x, 2 * y_1 - y])
        return_points.reverse()
        return_points += part_2[1:]  # reversed(part_2)
        #print('part_2 =', part_2)
        #print('result =', return_points)
        return return_points

    else:
        print("Облака совпали, разделение невозможно!")


# In[13]:


if __name__ == "__main__":

    object_1 = object_image()
    object_1.create_norm_distribution(
        N=1000,
        math_exp=[0, 15],
        cov_m=[[10, 0], [0, 20]]  # при 2 до -6; при 1 до -4
    )

    object_2 = object_image()
    object_2.create_norm_distribution(
        N=1000,
        math_exp=[30, 30],  # x, y стремятся к [25, 20]
        cov_m=[[50, 0], [0, 20]]  # по факту - разброс "вширь" по x и y [[50, 0], [0, 50]]
    )

    # object_1.show_points(10)
    build_graph_with_points(object_1, object_2)

    #object_1.show_3d_plot_of_distribution(10, 10)
    #print(object_1.return_probability(1,0))

    #from scipy.stats import norm
    #print(norm.cdf(0, 0, 15))
    #norm_distribution = \
    #sps.multivariate_normal.cdf([0,0], mean=[0,0], cov=[[1, 0], [0, 1]])
    #print(norm_distribution)


# In[ ]:


# #####

# In[ ]: