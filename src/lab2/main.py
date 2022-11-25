import math
import random
import itertools
import time
import sys

import scipy.stats as sps
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from numpy.core.records import ndarray
from sklearn.svm import SVC

np.random.seed(int(time.time()))


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
            'cov_lambda': None, # Функция (lambda ft_i, ft_j: 0) для вычисления Ковариация между i-признаком (данным) и j-признаком (где j=i+1)
            }
            Md_ft['y'] =
            'M': float, # математическое ожидание признака y
            'D': float, # дисперсия случайной величины признака y
            'cov_lambda': None, # Функция (lambda ft_i, ft_j: 0) для вычисления Ковариация между i-признаком (данным) и j-признаком (где j=i+1)
            }

            ...
            Md_ft['n'] =
            'M': float, # математическое ожидание признака n
            'D': float, # дисперсия случайной величины признака n
            'cov_lambda': None, # Функция (lambda ft_i, ft_j: 0) для вычисления Ковариация между i-признаком (данным) и j-признаком (где j=i+1)
            }

        """
        self._features_names = []
        self._dimensionality = 0
        self._size = N
        self._images = []
        self._default_cov = []

        for key, val in Md_ft.items():
            if not isinstance(key, (str, int)):
                raise ValueError('Наименование признака должно быть строкой или натуральным числом')
            if not isinstance(val, dict) or ('M' not in val) or ('D' not in val):
                raise ValueError(f'Параметры облака образа признака {key} должны быть словарем, содержащим как минимум ключи M и D')

            # все норм
            self._features_names.append(key)
            self._dimensionality += 1
            setattr(self, key, {**val})

        for i, fnamei in enumerate(self.features_names):
            cov_i_row = []
            fsetti = getattr(self, fnamei)
            for j, fnamej in enumerate(self.features_names):
                fsettj = getattr(self, fnamej)

                # Диагональ матрицы всегда дисперсия
                if i == j and fnamei == fnamej:
                    cov_i_row.append(fsettj['D'])

                # Ковариация между i-признаком и j-признаком
                else:
                    cov_lambda = fsetti.get('cov_lambda', None)
                    if cov_lambda and hasattr(cov_lambda, '__call__'):
                        cov_i_row.append(cov_lambda(fnamei, fnamej))
                    else:
                        cov_i_row.append(0)

            self._default_cov.append(cov_i_row)

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
        """
        Обычное заполнение облака образами с _независимым_ (получается одномерным) нормальным распределением каждого признака
        """
        del self._images
        self._images = []

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

    def fill_cloud_Rn_dimension(self):
        """
        Заполнение облака по нормальному распределению исходя из размерности облака
        """
        del self._images
        self._images = []

        true_dispersion = None

        features_Ms = []
        for key in self.features_names:
            fsett = getattr(self, key)
            if true_dispersion is None:
                true_dispersion = fsett['D']

            #===================================================================
            # if fsett['D'] != true_dispersion:
            #     raise ValueError('В режиме заливки "по полной размерности облака" необходимо равенство дисперсий каждого признака')
            #===================================================================

            features_Ms.append(fsett['M'])

        mean = features_Ms
        cov = self._default_cov

        *features_arrays, = np.random.multivariate_normal(mean, cov, self.size).T

        for features in itertools.zip_longest(*features_arrays):
            ftu = {k: v for k, v in itertools.zip_longest(self.features_names, features)}
            self._images.append(Image(**ftu))

        pass

    def pdf_Rn_dimension_scypy(self, x: Image):
        """
        Пло́тность вероя́тности (probability density function - PDF) - scypy

        :param x:
        """
        if x.dimensionality != self.dimensionality:
            raise ValueError("Размерность образа и облака не соотносятся")

        cov_m = self._default_cov
        mu = [getattr(self, f)['M'] for f in self.features_names]
        norm_distribution = sps.multivariate_normal(mean=mu, cov=cov_m)
        features_value = [getattr(x, f) for f in self.features_names]
        return norm_distribution.pdf(np.array(features_value))

    def pdf_Rn_dimension(self, x: Image):
        """
        Пло́тность вероя́тности (probability density function - PDF)

        :param x:
        """
        if x.dimensionality != self.dimensionality:
            raise ValueError("Размерность образа и облака не соотносятся")

        sigma = np.matrix(self._default_cov)
        mu = np.array([getattr(self, f)['M'] for f in self.features_names])

        size = x.dimensionality

        if size == len(mu) and (size, size) == sigma.shape:
            features_value = np.array([getattr(x, f) for f in self.features_names])

            det = np.linalg.det(sigma)  # Детерминант
            if det == 0:
                raise ValueError("The covariance matrix can't be singular")

            norm_const = 1.0 / (math.pow((2 * np.pi), float(size) / 2) * math.pow(det, 1.0 / 2))
            x_mu = np.matrix(features_value - mu)
            inv = sigma.I
            result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
            return norm_const * result
        else:
            raise ValueError("Размерность образа и ковариационной матрицы не соотносятся")

        pass

    def get_feature_iterator(self, feature_name):
        for im in self._images:
            yield getattr(im, feature_name)

    def get_feature_list(self, feature_name):
        return list(self.get_feature_iterator(feature_name))


class CloudComparator:
    """
    Сравнитель облак образов
    """
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

        # Обрежем размерность до R2
        fe_r2 = self.features_names[:2]
        if len(fe_r2) != len(self.features_names) != 2:
            raise ValueError('Размерность облаков не равна 2')

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
        if len(fe_r2) != len(self.features_names) != 2:
            raise ValueError('Размерность облаков не равна 2')

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
    cloud1 = ClassNormalCloud(100, x={'M': 800, 'D': 10000}, y={'M': 1200, 'D': 10000})
    cloud1.fill_cloud_Rn_dimension()

    print(cloud1.pdf_Rn_dimension_scypy(cloud1._images[0]))
    print(cloud1.pdf_Rn_dimension(cloud1._images[0]))

    cloud2 = ClassNormalCloud(100, x={'M': 1300, 'D': 10000}, y={'M': 1300, 'D': 10000})
    cloud2.fill_cloud_Rn_dimension()

    features_x1 = list(itertools.chain(cloud1.get_feature_iterator('x')))
    features_y1 = list(itertools.chain(cloud1.get_feature_iterator('y')))

    features_x2 = list(itertools.chain(cloud2.get_feature_iterator('x')))
    features_y2 = list(itertools.chain(cloud2.get_feature_iterator('y')))

    # Построение Координатной плоскости облака образов
    fig, ax = plt.subplots(figsize=(10, 6), num='Облака образов')

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
    # / Координаты середины отрезка

    # Координаты точка отрезка соединяющего середину и перпендикуляр
    normal_point = comparator.get_normal_image_r2_analityc()
    lnorm = mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x")
    ax.add_line(lnorm)
    # / Координаты точка отрезка соединяющего середину и перпендикуляр

    # ========================
    # Program Body
    # ========================

    # ========================
    # / Program Body
    # ========================

    plt.show()
    sys.exit()
