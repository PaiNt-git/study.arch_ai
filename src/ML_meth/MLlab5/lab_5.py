#cell 0

# Лаба 5, вариант 8

[Видосик](https://www.youtube.com/watch?v=PDCLoE-xOWM&list=PLA0M1Bcd0w8zxDIDOTQHsX68MCDOAJDtj&index=20)
[код на github](https://github.com/selfedu-rus/machine_learning/blob/main/machine_learning_20_1_svm.py)
[код на github 2](https://github.com/selfedu-rus/machine_learning/blob/main/machine_learning_20_2_svm.py)


#cell 1
import numpy as np  # 1.23.4
import ipywidgets as widgets  # 8.0.2
import matplotlib.pyplot as plt  # 3.6.1
from copy import deepcopy
from sklearn import svm
# использовать системное приложение для взаимодействия с графиками
# в моем случае - TkAgg
%matplotlib

#cell 2
if __name__ == "__main__":

    debug = True

    data_x = [(2.9, 6.0), (3.8, 5.1), (3.0, 4.9), (3.5, 5.0), (2.6, 5.5), (3.4, 4.6), (3.8, 5.1), (3.5, 5.5), (2.3, 5.0), (3.6, 4.9), (3.5, 5.1), (2.8, 5.7), (3.0, 5.4), (2.9, 6.4), (3.0, 4.3), (3.0, 4.8), (3.5, 5.1), (3.2, 4.7), (2.8, 5.7), (4.2, 5.5), (2.5, 6.3), (2.4, 4.9), (3.1, 4.8), (3.7, 5.4), (3.0, 5.6), (2.7, 5.6), (3.1, 6.9), (2.7, 6.0), (3.4, 4.8), (2.4, 5.5), (3.3, 5.1), (2.5, 5.6), (2.9, 6.2), (3.0, 5.9), (2.8, 6.1), (3.0, 4.4), (2.7, 5.2), (2.9, 5.7), (3.3, 5.0), (3.2, 6.4), (3.4, 5.2), (3.4, 5.0), (3.1, 4.9), (4.4, 5.7), (2.8, 6.1), (3.4, 5.0), (3.1, 6.7), (3.7, 5.1), (3.1, 4.9), (4.0, 5.8), (2.3, 4.5), (3.1, 6.7), (3.2, 5.0), (2.4, 5.5), (3.6, 5.0), (3.9, 5.4), (3.5, 5.0), (2.6, 5.7), (2.8, 6.8), (3.9, 5.4), (2.2, 6.0), (3.2, 4.4), (3.8, 5.7), (3.2, 4.7), (2.9, 6.6), (3.0, 4.8), (2.6, 5.8), (3.0, 5.0), (3.4, 5.1), (3.8, 5.1), (2.3, 6.3), (3.6, 4.6), (2.7, 5.8), (2.9, 4.4), (3.2, 4.6), (3.5, 5.2), (3.1, 4.6), (2.5, 5.5), (2.2, 6.2), (3.2, 7.0), (3.3, 6.3), (3.0, 6.1), (3.4, 4.8), (3.4, 5.4), (2.3, 5.5), (2.5, 5.1), (3.4, 6.0), (2.0, 5.0), (2.9, 5.6), (2.7, 5.8), (2.8, 6.5), (3.4, 5.4), (3.7, 5.3), (4.1, 5.2), (3.0, 5.6), (3.0, 6.6), (2.9, 6.1), (3.0, 6.7), (3.0, 5.7), (3.2, 5.9)]
    data_y = [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1]

    clean_data = list(set(zip(data_x, data_y)))  # чистим данные от дублей
    classes = {
        'green': {
            'data': np.array([list(data[0]) + [1] for data in clean_data if data[1] == -1]),
            'label': 'Образы 1 класса',
            'alpha': 1,
        },
        'blue': {
            'data': np.array([list(data[0]) + [1] for data in clean_data if data[1] == 1]),
            'label': 'Образы 2 класса',
            'alpha': 1,
        }
    }

#cell 3
    x_train = np.concatenate((classes['green']['data'], classes['blue']['data']))
    y_train = np.concatenate(
        (
            np.full(classes['green']['data'].shape[0], -1),
            np.full(classes['blue']['data'].shape[0], 1)
        )
    )
                                                 # regularization parameter
    clf = svm.SVC(kernel = 'linear',  gamma=0.7, C=1.0 )  # SVM с линейным ядром
    clf.fit(x_train, y_train)  # нахождение вектора w по обучающей выборке

    v = clf.support_vectors_  # выделение опорных векторов
    w = clf.coef_[0]
    # Перехват (также известный как смещение)
    # добавлен в функцию принятия решения. (тета 0)
    w0 = clf.intercept_
    # координаты разделяющей линии по осям
    dividing_line_xx = [np.min(x_train[:, 0]), np.max(x_train[:, 0])]  # относительно оси 0 (х)
    dividing_line_yy = np.dot((-1./w[1]), (np.dot(w[0],dividing_line_xx) + w0))

    # print(f"Разделяющая линия = {w}", f"Опорные вектора = {v}", sep='\n')

    y_pr = clf.predict(x_train)  # проверка на обучающей выборке
    # нули - без ошибок; иначе - ошибка
    number_of_errors = x_train.shape[0] - np.count_nonzero((np.array(y_train) - np.array(y_pr))==0) # .count(0)
    error_rate = 100*number_of_errors / x_train.shape[0]

#cell 4
    # Построение графиков ----------------------------------------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    fig.suptitle(f'SVM\n(вариант №8, размер выборки: {len(clean_data)})', fontsize=16)

    for class_i in classes:
        ax.scatter(
            classes[class_i]['data'][:, :1],
            classes[class_i]['data'][:, 1:2],
            color=class_i, label=classes[class_i]['label'],
            alpha=classes[class_i]['alpha'],
        )

    ax.scatter(
        v[:, 0], v[:, 1],
        s=150, edgecolor=None,
        alpha=0.5, color='red',
        linewidths=1, marker='*',
        label='Точки опорного вектора'
    )

    ax.plot(dividing_line_xx, dividing_line_yy, color='orange')

    ax.tick_params(labelcolor='indigo')
    ax.legend()
    ax.set_title(
        'Распределение классификатора: ' +
        f"{number_of_errors} ошибок = " +
        f"{error_rate} %",
        color='black'
    )

    plt.show()
    # ------------------------------------------------------------------------------

#cell 5


#cell 6


