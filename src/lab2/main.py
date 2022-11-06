import math
import random
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from numpy.core.records import ndarray


np.random.seed(42)


class Image:
    """
    Образ с произвольным набором признаков
    """

    def __init__(self, **features):
        """
        features: распаковка набора признаков со значениями, значения должные быть действительными числами
        """
        for key, val in features.items():
            if not isinstance(val, (int, float, ndarray)):
                raise ValueError('Значения признаков должны быть действительными числами')
            setattr(self, key, val)

        self._dimensionality = len(features)
        self._features_names = features.keys()

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def features_names(self):
        return self._features_names


class ClassNormalCloud:
    """
    Облако образов в пространстве признаков
    """

    def __init__(self, N, **Md_ft):
        """
        :param N: размер облака

        Md_ft: распаковка параметров для нормального распределения признаков
            Md_ft['x'] =
            'M': float, # математическое ожидание признака x
            'D': float, # дисперсия случайной величины признака x
            }
            Md_ft['y'] =
            'M': float, # математическое ожидание признака y
            'D': float, # дисперсия случайной величины признака y
            }

            ...
            Md_ft['n'] =
            'M': float, # математическое ожидание признака n
            'D': float, # дисперсия случайной величины признака n
            }

        """
        self._features_names = []
        self._dimensionality = 0
        self._size = N
        self._images = []

        for key, val in Md_ft.items():
            if not isinstance(key, (str, int)):
                raise ValueError('Наименование признака должно быть строкой или натуральным числом')
            if not isinstance(val, dict) or ('M' not in val) or ('D' not in val):
                raise ValueError(f'Параметры облака образа признака {key} должны быть словарем, содержащим как минимум ключи M и D')

            # все норм
            self._features_names.append(key)
            self._dimensionality += 1
            setattr(self, key, {**val})

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def features_names(self):
        return self._features_names

    @property
    def size(self):
        return self._size

    def fill_cloud(self):

        features_appropriate = []
        for key in self.features_names:
            fsett = getattr(self, key)
            sko = math.sqrt(fsett['D'])
            features_appropriate.append(np.nditer(np.random.normal(fsett['M'], sko, self.size)))  # fsett['M']-матожидание величины признака sko-СКО  self.size.-размер массива

        for i, features in enumerate(itertools.zip_longest(*features_appropriate)):
            ftu = {k: v for k, v in itertools.zip_longest(self.features_names, features)}
            self._images.append(Image(**ftu))
            pass

        pass

    def get_feature_iterator(self, feature_name):
        for im in self._images:
            yield getattr(im, feature_name)

    def get_feature_list(self, feature_name):
        return list(self.get_feature_iterator(feature_name))


if __name__ == "__main__":
    cloud1 = ClassNormalCloud(100, x={'M': 800, 'D': 6000}, y={'M': 1200, 'D': 6000})
    cloud1.fill_cloud()

    cloud2 = ClassNormalCloud(100, x={'M': 1300, 'D': 6000}, y={'M': 1300, 'D': 6000})
    cloud2.fill_cloud()

    features_x1 = list(itertools.chain(cloud1.get_feature_iterator('x')))
    features_y1 = list(itertools.chain(cloud1.get_feature_iterator('y')))

    features_x2 = list(itertools.chain(cloud2.get_feature_iterator('x')))
    features_y2 = list(itertools.chain(cloud2.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.annotate(f'({cloud1.x["M"]},\n {cloud1.y["M"]})',
                (cloud1.x["M"], cloud1.y["M"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")
    ax.annotate(f'({cloud2.x["M"]},\n {cloud2.y["M"]})',
                (cloud2.x["M"], cloud2.y["M"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")

    # Координаты середины отрезка
    midx = ((cloud2.x['M'] + cloud1.x['M']) / 2)
    midy = ((cloud2.y['M'] + cloud1.y['M']) / 2)
    ax.plot(midx, midy, color="red", marker='o')
    ax.annotate(f'({midx},\n {midy})',
                (midx, midy),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='blue', backgroundcolor="#eae1e196")

    # Прочертим нормаль, найдем тангенс противолежащего угла к нормали
    dmy = (midy - cloud1.y['M'])
    tgnorm = np.round((cloud2.y['M'] - cloud1.y['M']) / (cloud2.x['M'] - cloud1.x['M']), 4)
    alpha = math.atan(tgnorm)
    midlen = (dmy / math.sin(alpha))
    norm = (midlen * tgnorm)

    normx1 = cloud1.x['M'] + math.sqrt((math.pow(norm, 2) + math.pow(midlen, 2)))
    normy1 = cloud1.y['M']

    normx2 = cloud2.x['M'] - math.sqrt((math.pow(norm, 2) + math.pow(midlen, 2)))
    normy2 = cloud2.y['M']

    lnorm = mlines.Line2D([normx1, normx2], [normy1, normy2], color="green", linestyle="-", marker="x")
    ax.add_line(lnorm)

    plt.show()
