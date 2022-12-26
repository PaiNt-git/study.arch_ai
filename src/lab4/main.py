import itertools
import math
import random
import sys
import time
import functools

from collections import defaultdict

from numpy import inf, nan, isnan
from numpy.core.records import ndarray
from sklearn.svm import SVC

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sps

from lab2.main import *


np.random.seed(int(time.time()))


class NNPointRecogniteModel:
    """
    Класс Нейронной сети для распознавания точки
    """

    def __init__(self, nn_key: str='main'):
        """
        Модель с определенным алиасом должна создаваться один раз
        """
        self.nn_key = nn_key


NEURAL_MODEL_REGISTRY = {}


class NNCloudComparator(CloudComparator):
    """
    Сравнитель облак образов (ИНС)
    """
    cloud1 = None
    cloud2 = None

    def classify_image_nn(self, testpoint: Image, nn_key: str='main'):
        if nn_key not in NEURAL_MODEL_REGISTRY:
            NEURAL_MODEL_REGISTRY[nn_key] = NNPointRecogniteModel(nn_key)

        self.nn = NEURAL_MODEL_REGISTRY[nn_key]

        return -1

    def add_to_dataset(self):
        self.nn.add_to_dataset()
        self.nn.add_to_dataset()


if __name__ == "__main__":

    print('Введите N1: ')
    N1 = input()
    N1 = int(N1) if N1 else 10000

    print('Введите N2: ')
    N2 = input()
    N2 = int(N2) if N2 else 10000

    print('Введите Mx1: ')
    Mx1 = input()
    Mx1 = int(Mx1) if Mx1 else 600

    print('Введите My1: ')
    My1 = input()
    My1 = int(My1) if My1 else 500

    print('Введите Mx2: ')
    Mx2 = input()
    Mx2 = int(Mx2) if Mx2 else 200

    print('Введите My2: ')
    My2 = input()
    My2 = int(My2) if My2 else 700

    print('Введите Dx1: ')
    Dx1 = input()
    Dx1 = int(Dx1) if Dx1 else 10000

    print('Введите Dy1: ')
    Dy1 = input()
    Dy1 = int(Dy1) if Dy1 else 8000

    print('Введите Dx2: ')
    Dx2 = input()
    Dx2 = int(Dx2) if Dx2 else 80000

    print('Введите Dy2: ')
    Dy2 = input()
    Dy2 = int(Dy2) if Dy2 else 1000

    print('Значения введены, программа расчитывает оптимальную линию...')

    cloud1 = ClassNormalCloud(N1, x={'M': Mx1, 'D': Dx1}, y={'M': My1, 'D': Dy1}, klass=2)
    cloud1.fill_cloud_Rn_dimension()

    cloud2 = ClassNormalCloud(N2, x={'M': Mx2, 'D': Dx2}, y={'M': My2, 'D': Dy2}, klass=1)
    cloud2.fill_cloud_Rn_dimension()

    features_x1 = list(itertools.chain(cloud1.get_feature_iterator('x')))
    features_y1 = list(itertools.chain(cloud1.get_feature_iterator('y')))

    features_x2 = list(itertools.chain(cloud2.get_feature_iterator('x')))
    features_y2 = list(itertools.chain(cloud2.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6), num='Множества образов')

    # Чтобы перпендикуляры были перпендикулярными
    ax.set_aspect('equal', adjustable='box')

    # Удаление верхней и правой границ
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Добавление основных линий сетки
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # Образы
    ax.scatter(features_x1, features_y1, color="#8C7298")
    ax.scatter(features_x2, features_y2, color="#be542ccc")

    # Линия соединяющие центры облаков
    lM = mlines.Line2D([cloud1.x['M'], cloud2.x['M']], [cloud1.y['M'], cloud2.y['M']], color="#000", linestyle="--", marker="x")
    ax.add_line(lM)
    ax.annotate(f'({cloud1.x["M"]},\n {cloud1.y["M"]}):{cloud1.klass}',
                (cloud1.x["M"], cloud1.y["M"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")
    ax.annotate(f'({cloud2.x["M"]},\n {cloud2.y["M"]}):{cloud2.klass}',
                (cloud2.x["M"], cloud2.y["M"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")

    # Сравнитель
    comparator = NNCloudComparator(cloud1, cloud2)

    # Координаты середины отрезка
    mid_point = comparator.mid_image
    mid_len = comparator.mid_len

    ax.plot(mid_point.x, mid_point.y, color="red", marker='o')
    ax.annotate(f'({mid_point.x},\n {mid_point.y})',
                (mid_point.x, mid_point.y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")
    # / Координаты середины отрезка

    # Координаты точка отрезка соединяющего середину и перпендикуляр
    normal_point = comparator.get_normal_image_r2_main('x', 'y')
    lnorm = mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x", linewidth=0.8, )
    ax.add_line(lnorm)
    # / Координаты точка отрезка соединяющего середину и перпендикуляр

    # ========================
    # Program Body
    # ========================

    # Тестирование точки. Подпись угла
    for testpoint in [random.choice(cloud1._images[:5]),
                      random.choice(cloud2._images[:5]),
                      random.choice(cloud1._images[-5:]),
                      random.choice(cloud2._images[-5:])]:

        testklass = comparator.classify_image_nn(testpoint)
        xtest = round(testpoint.x, 2)
        ytest = round(testpoint.y, 2)

        ax.add_line(
            mlines.Line2D(
                [mid_point.x, testpoint.x],
                [mid_point.y, testpoint.y],
                color="purple",
                linestyle='dotted',
                linewidth=0.8,
                marker="x")
        )
        ax.annotate(f'({xtest},\n{ytest}),\nкласс {testklass}',
                    (testpoint.x, testpoint.y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    color='#83017b', backgroundcolor="#cea7a7db",
                    )

    # ========================
    # / Program Body
    # ========================

    print(f'N1={N1}, N2={N2}\nMx1={Mx1}, My1={My1}, Dx1={Dx1}, Dy1={Dy1}\nMx2={Mx2}, My2={My2}, Dx2={Dx2}, Dy2={Dy2}')
    plt.title(f'N1={N1}, N2={N2}\nMx1={Mx1}, My1={My1}, Dx1={Dx1}, Dy1={Dy1}\nMx2={Mx2}, My2={My2}, Dx2={Dx2}, Dy2={Dy2}')

    plt.show()
    sys.exit()
