#!/usr/bin/env python
# coding: utf-8

#
# # Лаба 4, вариант 8
#
# [Видосик](https://www.youtube.com/watch?v=3yVaheFr6dc&list=PLA0M1Bcd0w8zxDIDOTQHsX68MCDOAJDtj&index=16)
# [Херовый код на github](https://github.com/selfedu-rus/machine_learning/blob/main/machine_learning_16.py)
#

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


# In[15]:


if __name__ == "__main__":

    debug = True

    data_x = [(2.9, 6.0), (3.8, 5.1), (3.0, 4.9), (3.5, 5.0), (2.6, 5.5), (3.4, 4.6), (3.8, 5.1), (3.5, 5.5), (2.3, 5.0), (3.6, 4.9), (3.5, 5.1), (2.8, 5.7), (3.0, 5.4), (2.9, 6.4), (3.0, 4.3), (3.0, 4.8), (3.5, 5.1), (3.2, 4.7), (2.8, 5.7), (4.2, 5.5), (2.5, 6.3), (2.4, 4.9), (3.1, 4.8), (3.7, 5.4), (3.0, 5.6), (2.7, 5.6), (3.1, 6.9), (2.7, 6.0), (3.4, 4.8), (2.4, 5.5), (3.3, 5.1), (2.5, 5.6), (2.9, 6.2), (3.0, 5.9), (2.8, 6.1), (3.0, 4.4), (2.7, 5.2), (2.9, 5.7), (3.3, 5.0), (3.2, 6.4), (3.4, 5.2), (3.4, 5.0), (3.1, 4.9), (4.4, 5.7), (2.8, 6.1), (3.4, 5.0), (3.1, 6.7), (3.7, 5.1), (3.1, 4.9), (4.0, 5.8), (2.3, 4.5), (3.1, 6.7), (3.2, 5.0), (2.4, 5.5), (3.6, 5.0), (3.9, 5.4), (3.5, 5.0), (2.6, 5.7), (2.8, 6.8), (3.9, 5.4), (2.2, 6.0), (3.2, 4.4), (3.8, 5.7), (3.2, 4.7), (2.9, 6.6), (3.0, 4.8), (2.6, 5.8), (3.0, 5.0), (3.4, 5.1), (3.8, 5.1), (2.3, 6.3), (3.6, 4.6), (2.7, 5.8), (2.9, 4.4), (3.2, 4.6), (3.5, 5.2), (3.1, 4.6), (2.5, 5.5), (2.2, 6.2), (3.2, 7.0), (3.3, 6.3), (3.0, 6.1), (3.4, 4.8), (3.4, 5.4), (2.3, 5.5), (2.5, 5.1), (3.4, 6.0), (2.0, 5.0), (2.9, 5.6), (2.7, 5.8), (2.8, 6.5), (3.4, 5.4), (3.7, 5.3), (4.1, 5.2), (3.0, 5.6), (3.0, 6.6), (2.9, 6.1), (3.0, 6.7), (3.0, 5.7), (3.2, 5.9)]
    data_y = [1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1]

    clean_data = set(zip(data_x, data_y))  # чистим данные от дублей
    classes = {
        'green': {
            'data': np.array([data[0] for data in clean_data if data[1] == -1]),
            'label': 'Образы 1 класса',
            'alpha': 1,
        },
        'blue': {
            'data': np.array([data[0] for data in clean_data if data[1] == 1]),
            'label': 'Образы 2 класса',
            'alpha': 1,
        }
    }

    classes['green']['math_exp'] = {
        'x': np.mean(classes['green']['data'], axis=0)[0],
        'y': np.mean(classes['green']['data'], axis=0)[1],
    }
    classes['blue']['math_exp'] = {
        'x': np.mean(classes['blue']['data'], axis=0)[0],
        'y': np.mean(classes['blue']['data'], axis=0)[1],
    }

    classes['green']['dispersion'] = {
        'x': np.var(classes['green']['data'], axis=0, ddof=1)[0],
        'y': np.var(classes['green']['data'], axis=0, ddof=1)[1],
    }
    classes['blue']['dispersion'] = {
        'x': np.var(classes['blue']['data'], axis=0, ddof=1)[0],
        'y': np.var(classes['blue']['data'], axis=0, ddof=1)[1],
    }

    if debug:
        print('Мат. ожидание:')
        print('Зеленые:', classes['green']['math_exp'])
        print('Синие:', classes['blue']['math_exp'])

        print('\nДисперсия:')
        print('Зеленые:', classes['green']['dispersion'])
        print('Синие:', classes['blue']['dispersion'])


# In[16]:


# определение классификатора ---------------------------------------------------
classes['green']['classifier'] = \
    lambda x: -(x[0] - classes['green']['math_exp']['x']) ** 2 / (2 * classes['green']['dispersion']['x']) - (x[1] - classes['green']['math_exp']['y']) ** 2 / (2 * classes['green']['dispersion']['y'])

classes['blue']['classifier'] = \
    lambda x: -(x[0] - classes['blue']['math_exp']['x']) ** 2 / (2 * classes['blue']['dispersion']['x']) - (x[1] - classes['blue']['math_exp']['y']) ** 2 / (2 * classes['blue']['dispersion']['y'])


def classifier(xy_value):
    # удаленность от мат. ожидания в сочитание со знанием дисперсии
    # позволяет судить о вероятности принадлежности к классу
    # ("probably" - условное наименование в "", а не вероятность)
    green_probably = classes['green']['classifier'](xy_value)
    blue_probably = classes['blue']['classifier'](xy_value)
    if green_probably > blue_probably:
        return 'green', {'green': green_probably, 'blue': blue_probably}
    return 'blue', {'green': green_probably, 'blue': blue_probably}
# ------------------------------------------------------------------------------


# определяем точки с помощью классификатора для оценки точности ----------------
classes['green']['classifier_data'] = {
    'good_defined': [],
    'bad_defined': [],
    'bad_color': 'red',
}
classes['blue']['classifier_data'] = {
    'good_defined': [],
    'bad_defined': [],
    'bad_color': 'orange',
}

count_of_bad_defined = 0
for class_i in classes:
    for point in classes[class_i]['data']:
        if classifier(point)[0] == class_i:
            classes[class_i]['classifier_data']['good_defined'].append(point)
        else:
            classes[class_i]['classifier_data']['bad_defined'].append(point)
            count_of_bad_defined += 1
# ------------------------------------------------------------------------------

# Построение графиков ----------------------------------------------------------
for class_i in classes:

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    fig.suptitle(f'Наивный байесовский классификатор\n(вариант №8, размер выборки: {len(clean_data)})', fontsize=16)

    ax[0].set_title('Первоначальное распределение', color='black')

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
