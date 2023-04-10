"""
Лабораторная работа No4
Реализация наивного байесовского классификатора
Цель работы: научиться строить наивный байесовский классификатор и с его
помощью выполнять бинарную классификацию образов.

https://proproprogs.ru/ml/ml-bayesovskiy-vyvod-naivnaya-bayesovskaya-klassifikaciya
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import random


#===============================================================================
# Обучающая выборка
#===============================================================================
# вариант 5
data_x = [(5.8, 1.2), (5.6, 1.5), (6.5, 1.5), (6.1, 1.3), (6.4, 1.3), (7.7, 2.0), (6.0, 1.8), (5.6, 1.3), (6.0, 1.6), (5.8, 1.9), (5.7, 2.0), (6.3, 1.5), (6.2, 1.8), (7.7, 2.3), (5.8, 1.2), (6.3, 1.8), (6.0, 1.0), (6.2, 1.3), (5.7, 1.3), (6.3, 1.9), (6.7, 2.5), (5.5, 1.2), (4.9, 1.0), (6.1, 1.4), (6.0, 1.6), (7.2, 2.5), (7.3, 1.8), (6.6, 1.4), (5.6, 2.0), (5.5, 1.0), (6.4, 2.2), (5.6, 1.3), (6.6, 1.3), (6.9, 2.1), (6.8, 2.1), (5.7, 1.3), (7.0, 1.4), (6.1, 1.4), (6.1, 1.8), (6.7, 1.7), (6.0, 1.5), (6.5, 1.8), (6.4, 1.5), (6.9, 1.5), (5.6, 1.3), (6.7, 1.4), (5.8, 1.9), (6.3, 1.3), (6.7, 2.1), (6.2, 2.3), (6.3, 2.4), (6.7, 1.8), (6.4, 2.3), (6.2, 1.5), (6.1, 1.4), (7.1, 2.1), (5.7, 1.0), (6.8, 1.4), (6.8, 2.3), (5.1, 1.1), (4.9, 1.7), (5.9, 1.8), (7.4, 1.9), (6.5, 2.0), (6.7, 1.5), (6.5, 2.0), (5.8, 1.0), (6.4, 2.1), (7.6, 2.1), (5.8, 2.4), (7.7, 2.2), (6.3, 1.5), (5.0, 1.0), (6.3, 1.6), (7.7, 2.3), (6.4, 1.9), (6.5, 2.2), (5.7, 1.2), (6.9, 2.3), (5.7, 1.3), (6.1, 1.2), (5.4, 1.5), (5.2, 1.4), (6.7, 2.3), (7.9, 2.0), (5.6, 1.1), (7.2, 1.8), (5.5, 1.3), (7.2, 1.6), (6.3, 2.5), (6.3, 1.8), (6.7, 2.4), (5.0, 1.0), (6.4, 1.8), (6.9, 2.3), (5.5, 1.3), (5.5, 1.1), (5.9, 1.5), (6.0, 1.5), (5.9, 1.8)]
data_y = [-1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, -1, 1]
#===============================================================================


# Очистка данных ===============
# Удалим дубликаты
data_x_plus_y = zip(data_x, data_y)
data_x_plus_y = np.array([[*xy[0], xy[1]] for xy in data_x_plus_y])
data_x_plus_y = np.unique(data_x_plus_y, axis=0)
data_x = [list(xy)[:-1] for xy in data_x_plus_y]
data_y = [list(xy)[-1] for xy in data_x_plus_y]
# ==============================

# Синоним: -1 = c1, +1 = c2

x_train = np.array(data_x)
y_train = np.array(data_y)

# Матожидание
mx1_c1, mx2_c1 = np.mean(x_train[y_train == -1], axis=0)  # питон сахарок, предикат (y_train == 1) возвращает selector (array_like of bool) который может использоваться в __getitem__(). Это все считает матожидание от признаков определенного класса
mx1_c2, mx2_c2 = np.mean(x_train[y_train == 1], axis=0)

# Дисперсия
sx1_c1, sx2_c1 = np.var(x_train[y_train == -1], axis=0, ddof=1)   # формула для вычисления дисперсии здесь немного другая (1/N)*sum(...),
sx1_c2, sx2_c2 = np.var(x_train[y_train == 1], axis=0, ddof=1)  # Правильная формула дисперсии (1/(N-1))*sum((xi - mx)^2), Ddof нас спасет!

print(f'Класс -1, Матожидание признака mx1={mx1_c1}, mx2={mx2_c1}, ')
print(f'Класс +1, Матожидание признака mx1={mx1_c2}, mx2={mx2_c2}, ')

print(f'Класс -1, Дисперсия признака sx1={sx1_c1}, sx2={sx2_c1}, ')
print(f'Класс +1, Дисперсия признака sx1={sx1_c2}, sx2={sx2_c2}, ')


# модель на Плотность вероятности класса -1
def a_с1(x): return -(x[0] - mx1_c1) ** 2 / (2 * sx1_c1) - (x[1] - mx2_c1) ** 2 / (2 * sx2_c1)


# модель на Плотность вероятности класса +1
def a_с2(x): return -(x[0] - mx1_c2) ** 2 / (2 * sx1_c2) - (x[1] - mx2_c2) ** 2 / (2 * sx2_c2)


def classify(x):
    """
    Классификатор, -1, +1
    """
    return (-1 if int(np.argmax([a_с1(x), a_с2(x)])) == 0 else 1)


# Класифицированные образы
y_classified = [(i, a_с1(x), a_с2(x), classify(x), ) for i, x in enumerate(x_train)]


def _color_by_class(class_):
    if class_ == -1:
        return 'green'
    return 'blue'


# Построение графиков ----------------------------------------------------------
for class_ in (-1, 1):

    initial_class_points = x_train[y_train == class_]

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle(f'Наивный байесовский классификатор\n(вариант №5, размер выборки: {len(data_x)})', fontsize=16)

    ax[0].set_title('Первоначальное распределение', color='black')
    ax[0].scatter(
        initial_class_points.transpose()[0],
        initial_class_points.transpose()[1],
        color=_color_by_class(class_), label=f'Класс {class_}',
        alpha=1.0,
    )
    ax[0].tick_params(labelcolor='indigo')
    ax[0].legend()

    y_classified_predicate = [(True if (y_train[x[0]] == class_ and x[3] == class_) else False) for x in y_classified]
    classified_class_points = x_train[y_classified_predicate]

    y_not_classified_predicate = [(True if (y_train[x[0]] == class_ and x[3] != class_) else False) for x in y_classified]
    not_classified_class_points = x_train[y_not_classified_predicate]

    count_of_good_defined = len(classified_class_points)
    count_of_bad_defined = len(not_classified_class_points)

    ax[1].set_title("Распределение классификатора\n" +
                    f"{count_of_bad_defined} ошибок = " +
                    f"{100 * count_of_bad_defined / len(data_x)} %",
                    color='black'
                    )

    if count_of_good_defined:
        ax[1].scatter(
            classified_class_points.transpose()[0],
            classified_class_points.transpose()[1],
            color=_color_by_class(class_), label=f'Класс {class_}',
            alpha=1.0
        )

    if count_of_bad_defined:
        ax[1].scatter(
            not_classified_class_points.transpose()[0],
            not_classified_class_points.transpose()[1],
            color=_color_by_class(class_),
            label=f"Ошибки",
            alpha=0.2
        )

        for i, x in enumerate(not_classified_class_points):

            yc = y_classified[i]

            x_offset = round(random.uniform(-40.0, 40.0), 3)
            y_offset = round(random.uniform(-50.0, 60.0), 3)
            rand_rad = round(random.uniform(-1.0, 1.0), 1)

            ax[1].annotate(f'i={yc[0]}\nс1,c2={yc[1]:.3f},{yc[2]:.3f}',
                           (x[0], x[1]),
                           textcoords="offset points",
                           xytext=(0.0 + x_offset, 5.0 + y_offset),
                           ha='center',
                           color='#4e084a', backgroundcolor="#cea7a72b",
                           arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={rand_rad}")
                           )

    ax[1].tick_params(labelcolor='indigo')
    ax[1].legend()

    plt.show()
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
