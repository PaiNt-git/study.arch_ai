"""
Лабораторная работа №3
Исследование работы L2-регуляризатора в задачах регрессии
Цель работы: изучить особенности работы L2-регуляризатора на примере задачи
аппроксимации функции линейной моделью

"""

import numpy as np
import matplotlib.pyplot as plt


#===============================================================================
# Функция для исследования
#===============================================================================

# вариант 5

# x принадлежит множеству [0;10; 0,1]
x = np.arange(0, 10.1, 0.1)  # 10.1l для захватывания крайей точки


def y_from_x(x):  # функция в виде полинома -0.1x^5 + 5x^4 - 700x^2
    return (-0.1 * x**5 + 5 * x**4 - 700 * x**2)


y = np.array([y_from_x(x_) for x_ in x])

#===============================================================================


#===============================================================================
# Обучающая выборка
#===============================================================================

x_train, y_train = x[::2], y[::2]  # Четные точки - это обучающая

#===============================================================================


# Необходимо провести исследование полиноминальной функции


# Апроксимируем функцию полниномом Np степени
# Значения степени полинома ========================

Np = 12  # по заданию

# ==================================================


# numpy подбирает для обучающей выборки подходящие коэффициенты полиномов
z_train = np.polyfit(x_train, y_train, Np)

#===============================================================================
# numpy имеет встроенный генератор функции предикта по минимизации вектора ошибки
# она под капотом выглядит примерно так:
# def predict_poly(x, koeff):
#     res = 0
#     xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]
#     for i, k in enumerate(koeff):
#         res += k * xx[i]
#     return res
#===============================================================================
predict_poly_numpy = np.poly1d(z_train)

poly_koefs = z_train


y_func_repr = [np.round(yy, 5) for yy in y]
y_func_repr = y_func_repr[:3] + ['...'] + y_func_repr[-3:]

predict_numpy_repr = [np.round(np.poly1d(z_train)(xx), 5) for xx in x]
predict_numpy_repr = predict_numpy_repr[:3] + ['...'] + predict_numpy_repr[-3:]

print(f'Коэффициенты полинома: \n{poly_koefs}')
print(f'Исходные: y= {y_func_repr}')
print(f'Предсказание по {Np}-полиноминальной модели: y= {predict_numpy_repr}')

# размер признакового пространства (степень полинома Np+1) (+ w0)
N = Np + 1  # размер признакового пространства (степень полинома N-1)


# Т.к. мы используем L2 регуляризацию то производная функционала качества (по вектору w) будет иметь вид w* = (XT·X + lm·I)^-1 · XT·Y
# Поэтому в задаче стоит подбирать lm коэффициент, это и есть L2 регуляризация
# Значения коэфициента регуляризации ===============

lm = 21  # при увеличении N увеличивается lm (кратно): 12; 0.2   13; 20    15; 5000

# ==================================================


X = np.array([[a ** n for n in range(N)] for a in x])  # матрица входных векторов
lmI = np.array([[lm if i == j else 0 for j in range(N)] for i in range(N)])  # матрица lm*I, где I - единичная матрица
lmI[0][0] = 0  # первый коэффициент не регуляризуем (w0)
X_train = X[::2]  # обучающая выборка
Y = y_train  # обучающая выборка

# вычисление коэффициентов по формуле  w* = (XT·X + lm·I)^-1 · XT·Y
A = np.linalg.inv(X_train.T @ X_train + lmI)  # Инвариантная матрица
w = Y @ X_train @ A  # @ - 3-питоновский синтаксический сахар для операции numpy.matmul(), перемножение матриц
print(f'Весовые коээфициенты (отрегуляризированные): \n{w}')


# отображение исходного графика и прогноза
yy = [np.dot(w, x) for x in X]
plt.plot(x, yy, color='blue', label=f"Прогноз, по выборке (четные) с L2")  # Прогноз, обученный по выборке (четные) с L2 регуляризацией
plt.plot(x, y, color='red', label=f"Исходные значения всех точек")  # Исходные значения всех точек

plt.tick_params(labelcolor='indigo')
plt.legend()

plt.grid(True)
plt.show()


if __name__ == "__main__":
    pass
