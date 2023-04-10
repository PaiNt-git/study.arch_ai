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

print('Введите по какому алгоритму работать (selection - подбор, RMSE): ')
txt = ''
while not txt:
    try:
        line = input()
    except EOFError:
        break
    txt = line.strip()
    if not txt or txt.isspace():
        txt = 'RMSE'

w = [0, 0]


if txt == 'RMSE':

    # Аналитическое решение с помощью квадратичной функции ошибки
    pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)  # Сумма вектора x * y
    xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)  # Сумма x * xT
    invmat = np.linalg.inv(xxt)  # xxt^-1 или обратная матрица
    w = np.dot(pt, invmat)

    line_x = list(range(max(x_train[:, 0])))    # формирование графика разделяющей линии
    line_y = [-x * w[0] / w[1] - w[2] / w[1] for x in line_x]

else:

    # Параметры алгоритма (ступенчатая функция Хевисайда)===========================
    n_train = len(x_train)                          # размер обучающей выборки
    w = [0, -1, 0]                                  # начальное значение вектора w

    def a(x): return np.sign(x[0] * w[0] + x[1] * w[1])    # решающее правило (модель)

    N = 50                                          # максимальное число итераций
    nt = 0.1                                        # (эта) - шаг изменения веса
    e = 0.1                                         # небольшая добавка для w0 чтобы был зазор между разделяющей линией и граничным образом
    # ==============================================================================

    last_error_index = -1                        # индекс последнего ошибочного наблюдения
    for n in range(N):
        for i in range(n_train):                 # перебор по наблюдениям
            if y_train[i] * a(x_train[i]) < 0:   # если ошибка классификации (отступ M = y_train[i]*a(x_train[i])),
                w[0] = w[0] + nt * y_train[i]    # то корректировка веса w0
                last_error_index = i

        Q = sum([1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0])  # Функционал качества (list-comprehension) это нотация Айзерсона
        if Q == 0:      # показатель качества классификации (число ошибок), в общем случае → 0 для дифференцируемых функций. Но так как у нас кусочно-непрерывная
            break       # останов, если все верно классифицируем

    if last_error_index > -1:
        w[0] = w[0] + e * y_train[last_error_index]

    line_x = list(range(max(x_train[:, 0])))    # формирование графика разделяющей линии
    line_y = [w[0] * x for x in line_x]


def margin(x, y):
    """
    Отступы от линии разделения
    """
    return (x[0] * w[0] + x[1] * w[1] + w[2]) * y


margins = [margin(x, y) for x, y in zip(x_train, y_train)]


def classify(x):
    """
    Классификатор, -1, +1
    """
    return np.sign(x[0] * w[0] + x[1] * w[1] + w[2])


print(f'Весовые коэффициенты (вектор настраиваемых параметров): {list(w)}')
for i in range(len(x_train)):
    print(f'Точка {list(x_train[i][:-1])}, класс {y_train[i]}, отступ {margins[i]}')


x_0 = x_train[y_train == 1]                 # формирование точек для 1-го, numpy-python3 сахар, предикат в __getitem__()
x_1 = x_train[y_train == -1]                # и 2-го классов


line_sign = '-' if (-1 * (w[2] / w[1])) < 0 else '+'
abs_w2 = abs(w[2] / w[1])


plt.suptitle(f'Линейная бинарная классификация\n (вариант №5)\n ω = [{w[2]:.3f}, {w[1]:.3f}, {w[0]:.3f}]ᵀ', fontsize=12)
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
