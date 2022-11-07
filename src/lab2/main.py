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


class CloudComparator:
    cloud1 = None
    cloud2 = None

    def __init__(self, cloud1: ClassNormalCloud, cloud2: ClassNormalCloud):
        self.cloud1 = cloud1
        self.cloud2 = cloud2

        if self.cloud1.dimensionality != self.cloud2.dimensionality:
            raise ValueError('Размерность облаков образов не равна')

        if self.cloud1.features_names != self.cloud2.features_names:
            raise ValueError('Признаки облаков образов не совпадают')

        self._features_names = [x for x in self.cloud1.features_names]
        self._dimensionality = self.cloud1.dimensionality

    @property
    def dimensionality(self):
        return self._dimensionality

    @property
    def features_names(self):
        return self._features_names

    @staticmethod
    def get_between_point_len(image1: Image, image2: Image):
        if image1.dimensionality != image2.dimensionality:
            raise ValueError('Размерность образов не равна')

        if image1.features_names != image2.features_names:
            raise ValueError('Признаки образов не совпадают')

        quads = []
        for feature in image1.features_names:
            quads.append(math.pow((getattr(image2, feature) - getattr(image1, feature)), 2))

        return math.sqrt(sum(quads))

    @property
    def mid_image(self) -> Image:
        coords = {}
        for feature in self.features_names:
            mid_feature = ((getattr(self.cloud2, feature)['M'] + getattr(self.cloud1, feature)['M']) / 2)
            coords[feature] = mid_feature

        return Image(**coords)

    @property
    def mid_len(self):
        coords = {f: getattr(self.cloud1, f)['M'] for f in self.features_names}
        MImage = Image(**coords)
        ll = self.get_between_point_len(MImage, self.mid_image)
        return ll

    def get_normal_image_r2_main(self, znak='+', ) -> Image:
        mid_image = self.mid_image

        fe_r2 = self.features_names[:2]
        tgnorm_r2 = np.round((getattr(self.cloud2, fe_r2[1])['M'] - getattr(self.cloud1, fe_r2[1])['M']) / (getattr(self.cloud2, fe_r2[0])['M'] - getattr(self.cloud1, fe_r2[0])['M']), 4)
        norm_r2_len = (self.mid_len * tgnorm_r2)

        coords = {}

        featquad = sum([math.pow((getattr(self.cloud2, f)['M'] - getattr(self.cloud1, f)['M']), 2) for f in fe_r2])

        for i, feature in enumerate(fe_r2):

            if znak == '+':
                if i == 0:
                    coords[feature] = getattr(mid_image, fe_r2[0]) + norm_r2_len * ((getattr(self.cloud2, fe_r2[1])['M'] - getattr(self.cloud1, fe_r2[1])['M']) / math.sqrt(featquad))
                else:
                    coords[feature] = getattr(mid_image, fe_r2[1]) - norm_r2_len * ((getattr(self.cloud2, fe_r2[0])['M'] - getattr(self.cloud1, fe_r2[0])['M']) / math.sqrt(featquad))
            else:
                if i == 0:
                    coords[feature] = getattr(mid_image, fe_r2[0]) - norm_r2_len * ((getattr(self.cloud2, fe_r2[1])['M'] - getattr(self.cloud1, fe_r2[1])['M']) / math.sqrt(featquad))
                else:
                    coords[feature] = getattr(mid_image, fe_r2[1]) + norm_r2_len * ((getattr(self.cloud2, fe_r2[0])['M'] - getattr(self.cloud1, fe_r2[0])['M']) / math.sqrt(featquad))

        return Image(**coords)

    def get_normal_image_r2_analityc(self, znak='+', ) -> Image:

        # Обрежем размерность до R2
        fe_r2 = self.features_names[:2]

        # Координаты середины отрезка
        midx = ((getattr(self.cloud2, fe_r2[0])['M'] + getattr(self.cloud1, fe_r2[0])['M']) / 2)
        midy = ((getattr(self.cloud2, fe_r2[1])['M'] + getattr(self.cloud1, fe_r2[1])['M']) / 2)

        # Дельта y
        dmy = (midy - getattr(self.cloud2, fe_r2[1])['M'])

        # Найдем тангенс R2 противолежащего угла к нормали
        tgnorm_r2 = np.round((getattr(self.cloud2, fe_r2[1])['M'] - getattr(self.cloud1, fe_r2[1])['M']) / (getattr(self.cloud2, fe_r2[0])['M'] - getattr(self.cloud1, fe_r2[0])['M']), 4)

        # Длинна нормали
        alpha_r2 = math.atan(tgnorm_r2)
        midlen = (dmy / math.sin(alpha_r2))
        norm_r2_len = (midlen * tgnorm_r2)

        coords = {}

        if znak == '+':
            coords[fe_r2[0]] = getattr(self.cloud1, fe_r2[0])['M'] + math.sqrt((math.pow(norm_r2_len, 2) + math.pow(midlen, 2)))
            coords[fe_r2[1]] = getattr(self.cloud1, fe_r2[1])['M']
        else:
            coords[fe_r2[0]] = getattr(self.cloud2, fe_r2[0])['M'] - math.sqrt((math.pow(norm_r2_len, 2) + math.pow(midlen, 2)))
            coords[fe_r2[1]] = getattr(self.cloud1, fe_r2[1])['M']

        return Image(**coords)


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
    fig, ax = plt.subplots(figsize=(10, 6), num='Облака образов')

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

    # Сравнитель
    comparator = CloudComparator(cloud1, cloud2)

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

    # Координаты точка отрезка соединяющего середину и перпендикуляр
    normal_point = comparator.get_normal_image_r2_analityc()

    lnorm = mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x")
    ax.add_line(lnorm)

    plt.show()
