import itertools
import math
import random
import sys
import time

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier

from lab2.main import *


np.random.seed(int(time.time()))


class NNCloudComparator(CloudComparator):
    """
    Сравнитель облак образов (ИНС)
    """
    cloud1 = None
    cloud2 = None

    def __init__(self, cloud1: ClassNormalCloud, cloud2: ClassNormalCloud):
        super().__init__(cloud1=cloud1, cloud2=cloud2)

        self.nn_classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 2), random_state=1)

    def nn_fit(self):
        cloud1klass = self.cloud1.klass
        cloud2klass = self.cloud2.klass

        def ret_features_array(im):
            return np.array([getattr(im, x) for x in self.features_names])

        def ret_get_01_klass(im):
            klass = getattr(im, 'klass')
            if klass == cloud1klass:
                return 0
            elif klass == cloud2klass:
                return 1

        X = list(map(ret_features_array, self.cloud1._images + self.cloud2._images))
        y = list(map(ret_get_01_klass, (self.cloud1._images + self.cloud2._images)))
        self.nn_classifier.fit(X, y)

    def nn_get_recognite_images(self, cloud3, copy_image=False):

        cloud1klass = self.cloud1.klass
        cloud2klass = self.cloud2.klass

        def ret_features_array(im):
            return np.array([getattr(im, x) for x in self.features_names])

        def actual_class_fro_image(klass01):
            if klass01 == 0:
                return cloud1klass
            elif klass01 == 1:
                return cloud2klass

        imfeat = np.array(list(map(ret_features_array, cloud3._images)))
        recognition_result = self.nn_classifier.predict(imfeat)
        recognition_result = list(map(actual_class_fro_image, recognition_result))

        images = []
        for image, newklass in zip(cloud3._images, recognition_result):
            iminst = Image(**{f: getattr(f, image) for f in self.features_names}) if copy_image else image
            setattr(iminst, 'klass', newklass)
            images.append(iminst)

        return images


if __name__ == "__main__":

    print('Введите N1: ')
    N1 = input()
    N1 = int(N1) if N1 else 10000

    print('Введите N2: ')
    N2 = input()
    N2 = int(N2) if N2 else 10000

    print('Введите Mx1: ')
    Mx1 = input()
    Mx1 = int(Mx1) if Mx1 else 100

    print('Введите My1: ')
    My1 = input()
    My1 = int(My1) if My1 else 100

    print('Введите Mx2: ')
    Mx2 = input()
    Mx2 = int(Mx2) if Mx2 else 2500

    print('Введите My2: ')
    My2 = input()
    My2 = int(My2) if My2 else 110

    print('Введите Dx1: ')
    Dx1 = input()
    Dx1 = int(Dx1) if Dx1 else 10000

    print('Введите Dy1: ')
    Dy1 = input()
    Dy1 = int(Dy1) if Dy1 else 100000

    print('Введите Dx2: ')
    Dx2 = input()
    Dx2 = int(Dx2) if Dx2 else 10000

    print('Введите Dy2: ')
    Dy2 = input()
    Dy2 = int(Dy2) if Dy2 else 10000

    print('Значения введены, программа расчитывает оптимальную линию...')

    cloud1 = ClassNormalCloud(N1, x={'M': Mx1, 'D': Dx1}, y={'M': My1, 'D': Dy1}, klass=0)
    cloud1.fill_cloud_Rn_dimension()

    cloud2 = ClassNormalCloud(N2, x={'M': Mx2, 'D': Dx2}, y={'M': My2, 'D': Dy2}, klass=1)
    cloud2.fill_cloud_Rn_dimension()

    features_x1 = list(itertools.chain(cloud1.get_feature_iterator('x')))
    features_y1 = list(itertools.chain(cloud1.get_feature_iterator('y')))

    features_x2 = list(itertools.chain(cloud2.get_feature_iterator('x')))
    features_y2 = list(itertools.chain(cloud2.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6), num='Обучающие множества')

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

    print(f'\nОбучающие множества\nN1={N1}, N2={N2}\nMx1={Mx1}, My1={My1}, Dx1={Dx1}, Dy1={Dy1}\nMx2={Mx2}, My2={My2}, Dx2={Dx2}, Dy2={Dy2}')
    plt.title(f'Обучающие множества\nN1={N1}, N2={N2}\nMx1={Mx1}, My1={My1}, Dx1={Dx1}, Dy1={Dy1}\nMx2={Mx2}, My2={My2}, Dx2={Dx2}, Dy2={Dy2}')
    plt.show()

    # Получим минимальные и максимальные значения x, y
    x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = comparator.get_x_y_min_max_and_m('x', 'y')
    d_x = (x_max - x_min)
    x_mid = (x_max + x_min) / 2
    d_y = (y_max - y_min)
    y_mid = (y_max + y_min) / 2
    Dx = (abs(d_x) / 2)**2
    Dy = (abs(d_y) / 2)**2
    mid_D = (Dx + Dy) / 2
    mid_s = math.sqrt(mid_D)
    Sx = math.sqrt(Dx)
    Sy = math.sqrt(Dx)
    Dx_95 = (Sx * 3)**2
    Dy_95 = (Sy * 3)**2

    cloud3 = ClassNormalCloud(int((abs(d_x) * abs(d_y)) // (mid_s // 2)), x={'M': x_mid, 'D': Dx_95}, y={'M': y_mid, 'D': Dy_95})
    cloud3.fill_cloud_Rn_dimension()

    def uniform_filter(im):
        if im.x > x_max + Sx:
            return False
        elif im.x < x_min - Sx:
            return False
        if im.y < y_min - Sy:
            return False
        elif im.y > y_max + Sy:
            return False
        return True
    cloud3._images = list(filter(uniform_filter, cloud3._images))  # в области 3сигм больше всего значений. отсекаем аномалии

    # Образы равномерно по плоскости
    features_x3 = list(itertools.chain(cloud3.get_feature_iterator('x')))
    features_y3 = list(itertools.chain(cloud3.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6), num='Равномерно распределенные образы')

    # Чтобы перпендикуляры были перпендикулярными
    ax.set_aspect('equal', adjustable='box')

    # Удаление верхней и правой границ
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Добавление основных линий сетки
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    ax.scatter(features_x3, features_y3, color="#0b4b48")

    plt.title(f'Равномерно распределенные образы\n')
    plt.show()

    # Обучим сеть и распознаем облако
    comparator.nn_fit()
    comparator.nn_get_recognite_images(cloud3)

    # Образы классифицированные
    features_x3 = list(itertools.chain(cloud3.get_feature_iterator('x')))
    features_y3 = list(itertools.chain(cloud3.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6), num='Классифицированные образы')

    # Чтобы перпендикуляры были перпендикулярными
    ax.set_aspect('equal', adjustable='box')

    # Удаление верхней и правой границ
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Добавление основных линий сетки
    ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

    # Обучающие выборки
    ax.add_line(mlines.Line2D([cloud1.x['M'], cloud2.x['M']], [cloud1.y['M'], cloud2.y['M']], color="#000", linestyle="--", marker="x"))
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
    ax.plot(mid_point.x, mid_point.y, color="red", marker='o')
    ax.annotate(f'({mid_point.x},\n {mid_point.y})',
                (mid_point.x, mid_point.y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")
    normal_point = comparator.get_normal_image_r2_main('x', 'y')
    ax.add_line(mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x", linewidth=0.8, ))
    # / Обучающие выборки

    for image_ in cloud3._images:
        if image_.klass == cloud1.klass:
            color = 'red'
        elif image_.klass == cloud2.klass:
            color = 'blue'
        else:
            color = 'silver'
        ax.scatter(image_.x, image_.y, color=color)

    # Тестирование точки. Подпись
    for testpoint in [random.choice(cloud3._images[:5]),
                      random.choice(cloud3._images[-5:])]:

        testklass = testpoint.klass
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

    plt.title(f'Классифицированные образы\n(Обучающие множества: N1={N1}, N2={N2}\nMx1={Mx1}, My1={My1}, Dx1={Dx1}, Dy1={Dy1}\nMx2={Mx2}, My2={My2}, Dx2={Dx2}, Dy2={Dy2})')
    plt.show()
    sys.exit()
