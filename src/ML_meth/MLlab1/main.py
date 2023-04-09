"""
Лабораторная работа No1
Расчет коэффициентов разделяющей линии и вычисление отступа (margin)
для объектов разных классов
Цель работы: научиться вычислять коэффициенты разделяющей линии и
величину отступа (margin) при бинарной классификации объектов.

https://proproprogs.ru/ml/ml-funkcii-poter-v-zadachah-lineynoy-binarnoy-klassifikacii
"""

import numpy as np
import matplotlib.pyplot as plt


#===============================================================================
# Обучающая выборка
#===============================================================================

# вариант 5

x_train = [[1, 3], [7, 4], [4, 3], [9, 4]]
y_train = [-1, -1, 1, 1]  # -1 - зеленые, 1 - синие

#===============================================================================

# Добавим 1 для симметрии признаков и w
x_train = [list(x) + [1] for x in x_train]

x_train = np.array(x_train)
y_train = np.array(y_train)

pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)  # Сумма вектора x * y
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)  # Сумма x * xT
invmat = np.linalg.inv(xxt)  # xxt^-1 или обратная матрица
w = np.dot(pt, invmat)

print(f'Весовые коэффициенты (вектор настраиваемых параметров): {w}')


def classify(x):
    """
    Классификатор, -1, +1

    :param x:
    """
    asig = np.sign(x[0] * w[0] + x[1] * w[1] + w[2])

    return asig


line_x = list(range(max(x_train[:, 0])))    # формирование графика разделяющей линии
line_y = [-x * w[0] / w[1] - w[2] / w[1] for x in line_x]


x_0 = x_train[y_train == 1]                 # формирование точек для 1-го, numpy-python3 сахар, предикат в __getitem__()
x_1 = x_train[y_train == -1]                # и 2-го классов


line_sign = '-' if (-1 * (w[2] / w[1])) < 0 else '+'
abs_w2 = abs(w[2] / w[1])


plt.scatter(x_0[:, 0], x_0[:, 1], color='green', label=f"C1=-1")  # [:, 0] - питон 3 магия __getitem__ которую юзает numpy, эта запись значит "взять срез, тоесть копировать" и взять 0-й столбец из матрицы (первый)
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue', label=f"C2=+1")
plt.plot(line_x, line_y, color='red', label=f'Разделяющая линия,  x2 = {0-w[0]/w[1]:.3f}*x1 {line_sign} {abs_w2:.3f}')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("x2")
plt.xlabel("x1")

plt.tick_params(labelcolor='indigo')
plt.legend()

plt.grid(True)
plt.show()

if __name__ == "__main__":
    pass
