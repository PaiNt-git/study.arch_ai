"""
Лабораторная работа No4
Реализация наивного байесовского классификатора
Цель работы: научиться строить наивный байесовский классификатор и с его
помощью выполнять бинарную классификацию образов.

"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import chain


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
mx1_c1, mx2_c1 = np.mean(x_train[y_train == 1], axis=0)  # питон сахарок, предикат (y_train == 1) возвращает selector (array_like of bool) который может использоваться в __getitem__(). Это все считает матожидание от признаков определенного класса
mx1_c2, mx2_c2 = np.mean(x_train[y_train == -1], axis=0)

# Дисперсия
sx1_c1, sx2_c1 = np.var(x_train[y_train == 1], axis=0, ddof=1)   # формула для вычисления дисперсии здесь немного другая (1/N)*sum(...),
sx1_c2, sx2_c2 = np.var(x_train[y_train == -1], axis=0, ddof=1)  # Правильная формула дисперсии (1/(N-1))*sum((xi - mx)^2), Ddof нас спасет!

print(f'Класс -1, Матожидание признака mx1={mx1_c1}, mx2={mx2_c1}, ')
print(f'Класс +1, Матожидание признака mx1={mx1_c2}, mx2={mx2_c2}, ')

print(f'Класс -1, Дисперсия признака sx1={sx1_c1}, sx2={sx2_c1}, ')
print(f'Класс +1, Дисперсия признака sx1={sx1_c2}, sx2={sx2_c2}, ')


# модель на Плотность вероятности класса -1
def a_с1(x): return -(x[0] - mx2_c1) ** 2 / (2 * sx2_c1) - (x[1] - mx1_c1) ** 2 / (2 * sx1_c1)


# модель на Плотность вероятности класса +1
def a_с2(x): return -(x[0] - mx2_c2) ** 2 / (2 * sx2_c2) - (x[1] - mx1_c2) ** 2 / (2 * sx1_c2)


def classify(x): return (-1 if np.argmax([a_с1(x), a_с2(x)]) == 0 else 1)


# Класифицированные образы
y_classified = [(a_с1(x), a_с2(x), classify(x)) for x in x_train]


# Построение графиков ----------------------------------------------------------
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
fig.suptitle(f'Наивный байесовский классификатор\n(вариант №8, размер выборки: {len(clean_data)})', fontsize=16)

ax[0].set_title('Первоначальное распределение', color='black')

for class_i in classes:
    ax[0].scatter(
        classes[class_i]['data'].transpose()[0],
        classes[class_i]['data'].transpose()[1],
        color=class_i, label=classes[class_i]['label'],
        alpha=classes[class_i]['alpha'],
    )
    ax[0].tick_params(labelcolor='indigo')
    ax[0].legend()

    ax[1].set_title("Распределение классификатора\n" +
                    f"{count_of_bad_defined} ошибок = " +
                    f"{100 * count_of_bad_defined / len(clean_data)} %",
                    color='black'
                    )

    if classes[class_i]['classifier_data']['good_defined']:
        ax[1].scatter(
            np.array(classes[class_i]['classifier_data']['good_defined']).transpose()[0],
            np.array(classes[class_i]['classifier_data']['good_defined']).transpose()[1],
            color=class_i, label=classes[class_i]['label'],
            alpha=classes[class_i]['alpha']
        )

    if classes[class_i]['classifier_data']['bad_defined']:
        ax[1].scatter(
            np.array(classes[class_i]['classifier_data']['bad_defined']).transpose()[0],
            np.array(classes[class_i]['classifier_data']['bad_defined']).transpose()[1],
            color=classes[class_i]['classifier_data']['bad_color'],
            label="Ошибки " + classes[class_i]['label'],
            alpha=0.5
        )
    ax[1].tick_params(labelcolor='indigo')
    ax[1].legend()
    plt.show()
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
