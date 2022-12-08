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


np.random.seed(int(time.time()))


def memoize_func(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoizer


def memoize_method(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoizer(self, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        return cache[key]
    return memoizer


def k(p_1, p_2):
    # Уравнение линии (y = kx + b) по двум точкам p_1, p_2
    dzero = (p_2[0] - p_1[0])
    return np.float64((p_2[1] - p_1[1])) / dzero


def b(p_1, p_2):
    # Уравнение линии (y = kx + b) по двум точкам p_1, p_2
    dzero = (p_2[0] - p_1[0])
    return np.float64((p_1[1] * p_2[0] - p_1[0] * p_2[1])) / dzero


def pretty_line_x(b_, x_O, x):
    """
    При делении на 0, неопределенность, и нужны специальные занчения x

    :param b_:
    :param x_O:
    :param x:
    """
    if b_ in (-inf, inf) or isnan(b_):
        return x_O
    return x


def pretty_line_y(b_, y_min, y_max, y):
    """
    При делении на 0, неопределенность, и нужны специальные занчения y

    :param b_:
    :param y_min:
    :param y_max:
    :param y:
    """
    if b_ == -inf:
        return y_min
    elif b_ == inf or isnan(b_):
        return y_max
    return y


def rotate_line_equation(k_line, b_line, o_point, r_point, x_min, x_max, y_min, y_max, step_x=0):
    """
    Вернуть формулу (заивисимо от угла fi) которая вращаяется относительно o_point, r_point - точка на этой прямой

    :param k_line:
    :param b_line:
    :param o_point: (x, y)
    :param r_point: (x, y)
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param step_x:
    """
    if not step_x:
        step_x = abs((x_max - x_min) / 20)

    # Вращение прямой относительно точки пересечения
    # Общее уравнение прямой https://stackoverflow.com/questions/27442437/rotate-a-line-by-a-given-angle

    atanrad = math.atan(k_line)

    alfa = math.degrees(atanrad)

    if r_point[1] == o_point[1]:
        r_point = (o_point[0], y_max)

    lenline = math.sqrt(math.pow((r_point[0] - o_point[0]), 2) + math.pow((r_point[1] - o_point[1]), 2))

    def get_line_rotate_formula(fii):
        betta = (alfa - fii)
        betta = math.radians(betta)

        if fii != 0:
            x_rotate = lenline * math.cos(betta) + o_point[0]
            y_rotate = lenline * math.sin(betta) + o_point[1]

        k_rotate = k((o_point[0], o_point[1]), (x_rotate, y_rotate)) if fii != 0 else k_line
        b_rotate = b((o_point[0], o_point[1]), (x_rotate, y_rotate)) if fii != 0 else b_line

        def formul(xxx): return pretty_line_y(b_rotate, y_min, y_max, (k_rotate * xxx + b_rotate))

        if b_rotate not in (-inf, inf) and b_line not in (-inf, inf):

            # "Левая" часть, возрастание и убывание функций

            left_x = x_min
            left_y = formul(left_x)

            if left_y > y_max:
                while left_y > y_max:
                    left_x += step_x
                    left_y = formul(left_x)

            elif left_y < y_min:
                while left_y < y_min:
                    left_x += step_x
                    left_y = formul(left_x)

            if left_y < y_min:
                left_x = x_max
                left_y = formul(left_x)
                while left_y < y_min:
                    left_x -= step_x
                    left_y = formul(left_x)

            elif left_y > y_max:
                left_x = x_max
                left_y = formul(left_x)
                while left_y > y_max:
                    left_x += step_x
                    left_y = formul(left_x)

            # "Правая" часть, возрастание и убывание функций

            right_x = x_max
            right_y = formul(right_x)

            if right_y < y_min:
                while right_y < y_min:
                    right_x -= step_x
                    right_y = formul(right_x)

            elif right_y > y_max:
                while right_y > y_max:
                    right_x -= step_x
                    right_y = formul(right_x)

            if right_y > y_max:
                right_x = x_min
                right_y = formul(right_x)
                while right_y > y_max:
                    right_x += step_x
                    right_y = formul(right_x)

            elif right_y < y_min:
                right_x = x_min
                right_y = formul(right_x)
                while right_y < y_min:
                    right_x -= step_x
                    right_y = formul(right_x)

        else:
            left_x, right_x = lenline * math.cos(betta) + o_point[0], -lenline * math.cos(betta) + o_point[0]
            left_y, right_y = lenline * math.sin(betta) + o_point[1], -lenline * math.sin(betta) + o_point[1]

        k_rotate = k((left_x, left_y), (right_x, right_y)) if fii != 0 else k_line
        b_rotate = b((left_x, left_y), (right_x, right_y)) if fii != 0 else b_line

        return k_rotate, b_rotate, left_x, left_y, right_x, right_y

    return get_line_rotate_formula


def offset_line_equation_and_points(k_line, b_line, o_point, r_point, x_min, x_max, y_min, y_max, accuracy_step=0):
    """
    Смещение прямой относительно o_point, r_point вторая точка лежащая на прямой

    :param k_line:
    :param b_line:
    :param o_point: (x, y)
    :param r_point: (x, y)
    :param x_min:
    :param x_max:
    :param y_min:
    :param y_max:
    :param accuracy_step:
    """

    # Уравнение перепендикуляра к середине
    k_normal = k_line
    b_normal = b_line

    if not accuracy_step:
        accuracy_step = abs((x_max - x_min) / 50)

    def b_offset(xx, yy): return (yy * r_point[0] - xx * ((r_point[0] - xx) * k_normal + yy)) / (r_point[0] - xx)

    def get_line_offset_formula(step_x):

        _offset_x = o_point[0] + step_x
        b1 = b_offset(_offset_x, o_point[1])  # k_normal * left_x + b1

        def formul(xx): return pretty_line_y(b1, y_min, y_max, (k_normal * xx + b1))

        if b1 not in (-inf, inf) and b_line not in (-inf, inf):

            # "Левая" часть, возрастание и убывание функций

            left_x = x_min
            left_y = formul(left_x)

            if left_y > y_max:
                while left_y > y_max:
                    left_x += accuracy_step
                    left_y = formul(left_x)

            elif left_y < y_min:
                while left_y < y_min:
                    left_x += accuracy_step
                    left_y = formul(left_x)

            if left_y < y_min:
                left_x = x_max
                left_y = formul(left_x)
                while left_y < y_min:
                    left_x -= accuracy_step
                    left_y = formul(left_x)

            elif left_y > y_max:
                left_x = x_max
                left_y = formul(left_x)
                while left_y > y_max:
                    left_x += accuracy_step
                    left_y = formul(left_x)

            # "Правая" часть, возрастание и убывание функций

            right_x = x_max
            right_y = formul(right_x)

            if right_y < y_min:
                while right_y < y_min:
                    right_x -= accuracy_step
                    right_y = formul(right_x)

            elif right_y > y_max:
                while right_y > y_max:
                    right_x -= accuracy_step
                    right_y = formul(right_x)

            if right_y > y_max:
                right_x = x_min
                right_y = formul(right_x)
                while right_y > y_max:
                    right_x += accuracy_step
                    right_y = formul(right_x)

            elif right_y < y_min:
                right_x = x_min
                right_y = formul(right_x)
                while right_y < y_min:
                    right_x -= accuracy_step
                    right_y = formul(right_x)

        else:
            left_x, right_x = _offset_x, _offset_x
            left_y, right_y = y_min, y_max

        return k_normal, b1, left_x, left_y, right_x, right_y

    return get_line_offset_formula


class Image:
    """
    Образ с произвольным набором признаков
    """

    def __init__(self, klass=None, **features):
        """
        :param klass: Класс образа если задан

        features: распаковка набора признаков со значениями, значения должные быть действительными числами

        например можно вызывать класс Image(x=1, y=2)
        """
        self.klass = klass

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
    Множество образов в пространстве признаков
    """

    def __init__(self, N, klass=None, **Md_ft):
        """
        :param N: размер облака
        :param klass: Класс образа если задан

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
        self.klass = klass

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

    def fill_cloud_Rn_dimension(self):
        """
        Заполнение множества по нормальному распределению исходя из размерности облака
        """
        del self._images
        self._images = []

        true_dispersion = None

        features_Ms = []
        for key in self.features_names:
            fsett = getattr(self, key)
            if true_dispersion is None:
                true_dispersion = fsett['D']

            features_Ms.append(fsett['M'])

        mean = features_Ms
        cov = self._default_cov

        *features_arrays, = np.random.multivariate_normal(mean, cov, self.size).T

        for features in itertools.zip_longest(*features_arrays):
            ftu = {k: v for k, v in itertools.zip_longest(self.features_names, features)}
            self._images.append(Image(klass=self.klass, **ftu))

#===============================================================================
#     def pdf_Rn_dimension_from_destimator(self, x: Image, kernel_density_estimator):
#         """
#         Пло́тность вероя́тности (probability density function - PDF) - используя дестиматор
#
#         scipy.stats.gaussian_kde
#         KernelDensity
#
#         https://scikit-learn.org/stable/modules/density.html
#         https://stackoverflow.com/questions/52160088/python-fast-kernel-density-estimation-probability-density-function
#
#         from fastkde.fastKDE import pdf
#
#         def get_pdf(data):
#             y, x = pdf(data)
#             return x, y
#
#         :param x:
#         """
#         if x.dimensionality != self.dimensionality:
#             raise ValueError("Размерность образа и облака не соотносятся")
#
#         return None
#===============================================================================

    def pdf_Rn_dimension(self, x: Image):
        """
        Пло́тность вероя́тности (probability density function - PDF)
        Вернуть значение вероятности точки по ее признакам

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
                raise ValueError("Ковариационная матрица не может быть сингулярной")

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
        """
        Евклидоваое расстояние между точками

        :param image1:
        :param image2:
        """
        if image1.dimensionality != image2.dimensionality:
            raise ValueError('Размерность образов не равна')

        if image1.features_names != image2.features_names:
            raise ValueError('Признаки образов не совпадают')

        quads = []
        for feature in image1.features_names:
            quads.append(math.pow((getattr(image2, feature) - getattr(image1, feature)), 2))

        return math.sqrt(sum(quads))

    @memoize_method
    def get_x_y_min_max_and_m(self, x, y):
        """
        Вычилить максмумы один раз

        :param x:
        :param y:
        """

        # Мат. ожидания множеств
        x1_m = getattr(self.cloud1, x)['M']
        y1_m = getattr(self.cloud1, y)['M']
        x2_m = getattr(self.cloud2, x)['M']
        y2_m = getattr(self.cloud2, y)['M']

        # Вычислим минимальные и максимальные x
        x_min = min(itertools.chain(self.cloud1.get_feature_iterator(x), self.cloud2.get_feature_iterator(x)))
        x_min = x_min - x_min * 0.1

        x_max = max(itertools.chain(self.cloud1.get_feature_iterator(x), self.cloud2.get_feature_iterator(x)))
        x_max = x_max + x_max * 0.1

        y_min = min(itertools.chain(self.cloud1.get_feature_iterator(y), self.cloud2.get_feature_iterator(y)))
        y_min = y_min - y_min * 0.1

        y_max = max(itertools.chain(self.cloud1.get_feature_iterator(y), self.cloud2.get_feature_iterator(y)))
        y_max = y_max + y_max * 0.1

        return x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max

    @property
    @memoize_method
    def mid_image(self) -> Image:
        """
        Точка на середине отрезка соединяющего М множество

        """
        coords = {}
        for feature in self.features_names:
            mid_feature = ((getattr(self.cloud2, feature)['M'] + getattr(self.cloud1, feature)['M']) / 2)
            coords[feature] = mid_feature

        return Image(**coords)

    @property
    @memoize_method
    def mid_len(self):
        """
        Длина серединного отрезка

        """
        coords = {f: getattr(self.cloud1, f)['M'] for f in self.features_names}
        MImage = Image(**coords)
        ll = self.get_between_point_len(MImage, self.mid_image)
        return ll

    def get_normal_image_r2_main(self, feature_name1, feature_name2, znak='+', ) -> Image:
        """
        Координаты нормали к отрезку соединяющему множества

        :param feature_name1: какой признак взять за x
        :param feature_name2: какой признак взять за y
        :param znak:
        """
        mid_image = self.mid_image

        # Обрежем размерность до R2
        fe_r2 = [feature_name1, feature_name2]

        tgnorm_r2 = np.round((getattr(self.cloud2, fe_r2[1])['M'] - getattr(self.cloud1, fe_r2[1])['M']) / (getattr(self.cloud2, fe_r2[0])['M'] - getattr(self.cloud1, fe_r2[0])['M']), 4)
        norm_r2_len = (self.mid_len * tgnorm_r2)
        if not norm_r2_len:
            norm_r2_len = self.get_between_point_len(
                Image(**{feature_name1: getattr(self.cloud1, feature_name1)['M'], feature_name2: getattr(self.cloud1, feature_name2)['M']}),
                Image(**{feature_name1: getattr(self.cloud2, feature_name1)['M'], feature_name2: getattr(self.cloud2, feature_name2)['M']}),
            ) / 2

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

        # Заполним координаты недолстающих признаков нулями
        not_has_features = [x for x in self.features_names if x not in fe_r2]
        coords.update((k, 0) for k in not_has_features)

        return Image(**coords)

    def get_probability_circle_points(self, cloud_num=1, margin_of_error=0.0005, ax=None):
        """
        Нарисовать линию (окружность) разделения между множествами

        :param cloud_num: относительно какого множества рисовать
        :param margin_of_error:
        :param ax:
        """

        cloud1_features_value = [getattr(self.cloud1, f) for f in self.cloud1.features_names]
        cloud2_features_value = [getattr(self.cloud2, f) for f in self.cloud2.features_names]

        M1max = step_accuracy1 = max([x['M'] for x in cloud1_features_value])
        M2max = step_accuracy2 = max([x['M'] for x in cloud2_features_value])

        D1max = max(x['D'] for x in cloud1_features_value)
        margin_of_error1 = margin_of_error / D1max
        D2max = max(x['D'] for x in cloud2_features_value)
        margin_of_error2 = margin_of_error / D2max

        current_cloud = self.cloud1 if cloud_num == 1 else self.cloud2
        current_margin = margin_of_error1 if cloud_num == 1 else margin_of_error2
        current_step_accuracy = step_accuracy1 if cloud_num == 1 else step_accuracy2

        # Приведение гиперпространства к сумме двухмерных
        images = []
        circle_equations = {}

        for feature_r2 in itertools.combinations(current_cloud.features_names, 2):
            x, y = feature_r2
            x_m = getattr(current_cloud, x)['M']
            y_m = getattr(current_cloud, y)['M']

            not_has_features = [f for f in current_cloud.features_names if f not in (x, y)]

            # шаги изменений для алгоритма
            step_x = x_m / current_step_accuracy
            step_y = y_m / current_step_accuracy

            # определяем наибольшие точки сверху (смещение по y относительно М)
            _iter_counter = 0
            current_y = y_m + step_y * _iter_counter
            coords_top_y = {
                x: x_m,
                y: current_y,
            }
            coords_top_y.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)
            while abs(current_cloud.pdf_Rn_dimension(Image(**coords_top_y))) > current_margin:
                current_y = y_m + step_y * _iter_counter
                coords_top_y[y] = current_y
                _iter_counter += 1
            top_y = step_y * _iter_counter

            # определяем наибольшие точки справа (смещение по x относительно М)
            _iter_counter = 0
            current_x = x_m + step_x * _iter_counter
            coords_top_x = {
                x: current_x,
                y: y_m,
            }
            coords_top_x.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)
            while abs(current_cloud.pdf_Rn_dimension(Image(**coords_top_x))) > current_margin:
                current_x = x_m + step_x * _iter_counter
                coords_top_x[x] = current_x
                _iter_counter += 1
            top_x = step_x * _iter_counter

            if not (x, y) in circle_equations:
                _equations = {
                    'func_def': f'''{x} = A + B*cos(a)\n{y} = C + D*sin(a)''',
                    'A': x_m,
                    'B': top_x,
                    'C': y_m,
                    'D': top_y,
                    x: lambda a: x_m + top_x * math.cos(math.radians(a)),
                    y: lambda a: y_m + top_y * math.sin(math.radians(a)),
                }
                circle_equations[(x, y)] = _equations

            for al in range(0, 361, 30):
                x_ = circle_equations[(x, y)][x](al)
                y_ = circle_equations[(x, y)][y](al)

                coords_ = {
                    x: x_,
                    y: y_,
                }
                coords_.update((f, 0) for f in not_has_features)

                images.append(Image(**coords_))

        return images, circle_equations

    def get_probability_midlane_points(self, margin_of_error=0.0005, ax=None, fi=0, offset_from_O=0):
        """
        Нарисовать линию разделения между облаками на основе минимизации вероятности pdf

        :param cloud_num: относительно какого облака рисовать
        :param margin_of_error:
        :param ax:
        """

        cloud1_features_value = [getattr(self.cloud1, f) for f in self.cloud1.features_names]
        cloud2_features_value = [getattr(self.cloud2, f) for f in self.cloud2.features_names]

        M1max = step_accuracy1 = max([x['M'] for x in cloud1_features_value])
        M2max = step_accuracy2 = max([x['M'] for x in cloud2_features_value])

        D1max = max(x['D'] for x in cloud1_features_value)
        margin_of_error1 = margin_of_error / D1max
        D2max = max(x['D'] for x in cloud2_features_value)
        margin_of_error2 = margin_of_error / D2max

        # Приведение гиперпространства к сумме двухмерных
        images = []
        line_equations = {}
        cloud_plus_points = []
        cloud_minus_points = []

        step_x = 0.5
        step_y = 0.5

        # Вычислим минимальные и максимальные x
        x_min = 0
        x_max = 0
        y_min = 0
        y_max = 0

        for cloud_num in (1, 2):
            current_cloud = self.cloud1 if cloud_num == 1 else self.cloud2
            current_margin = margin_of_error1 if cloud_num == 1 else margin_of_error2
            current_step_accuracy = step_accuracy1 if cloud_num == 1 else step_accuracy2

            for feature_r2 in itertools.combinations(current_cloud.features_names, 2):
                x, y = feature_r2
                x_m = getattr(current_cloud, x)['M']
                y_m = getattr(current_cloud, y)['M']

                x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = self.get_x_y_min_max_and_m(x, y)

                not_has_features = [f for f in current_cloud.features_names if f not in (x, y)]

                # шаги изменений для алгоритма
                step_x = x_m / current_step_accuracy
                step_y = y_m / current_step_accuracy
                step_len = min((step_x, step_y))

                # определяем наибольшие точки относительно шага по линии

                # Уравнение линии соединения
                k_base = k((x1_m, y1_m), (x2_m, y2_m))
                b_base = b((x1_m, y1_m), (x2_m, y2_m))

                # Координаты середины отрезка
                mid_point = self.mid_image
                # Координаты точка отрезка соединяющего середину и перпендикуляр
                normal_point = self.get_normal_image_r2_main('x', 'y')

                # Уравнение перепендикуляра к середине
                k_normal = k((getattr(mid_point, x), getattr(mid_point, y)), (getattr(normal_point, x), getattr(normal_point, y)))
                b_normal = b((getattr(mid_point, x), getattr(mid_point, y)), (getattr(normal_point, x), getattr(normal_point, y)))

                if b_normal in (-inf, inf) or isnan(b_normal):
                    coords_1 = {
                        x: (x1_m + x2_m) / 2,
                        y: y_max,
                    }
                    coords_1.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)

                    coords_2 = {
                        x: (x1_m + x2_m) / 2,
                        y: y_min,
                    }
                    coords_2.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)

                    if not (x, y, 0) in line_equations:

                        _equations = {
                            'func_def': f'''{y} = {k_normal}*x + {b_normal}''',
                            'k': k_normal,
                            'b': b_normal,
                            f'{y}': lambda xx: pretty_line_y(b_normal, y_min, y_max, (k_normal * xx + b_top_plus)),
                            'center_point': self.mid_image,
                        }
                        line_equations[(x, y, 0)] = _equations

                        line_equations[(x, y, 0)][f'{x}_r_point'] = getattr(self.mid_image, x)
                        line_equations[(x, y, 0)][f'{y}_r_point'] = getattr(self.mid_image, y)

                        line_equations[(x, y, 0)][f'{x}_min'] = x_min
                        line_equations[(x, y, 0)][f'{x}_max'] = x_max
                        line_equations[(x, y, 0)][f'{y}_min'] = y_min
                        line_equations[(x, y, 0)][f'{y}_max'] = y_max
                        line_equations[(x, y, 0)][f'step_{x}'] = step_x

                        line_equations[(x, y, 0)][f'left_{x}'] = x_min
                        line_equations[(x, y, 0)][f'left_{y}'] = y_min
                        line_equations[(x, y, 0)][f'right_{x}'] = x_max
                        line_equations[(x, y, 0)][f'right_{y}'] = y_max

                        images.append(Image(**coords_1))
                        images.append(Image(**coords_2))
                        return images, line_equations

                else:
                    # Функция смещения для прямой
                    def b_frompoint_func(xx, yy): return (yy * getattr(normal_point, x) - xx * ((getattr(normal_point, x) - xx) * k_normal + yy)) / (getattr(normal_point, x) - xx)

                    # Перебор точек по линии (+)
                    _iter_counter = 0
                    current_x = x_m
                    current_y = y_m
                    coords_top_plus = {
                        x: current_x,
                        y: current_y,
                    }
                    coords_top_plus.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)
                    while abs(current_cloud.pdf_Rn_dimension(Image(**coords_top_plus))) > current_margin:
                        current_x = current_x + step_len * _iter_counter
                        current_y = pretty_line_y(b_base, y_min, y_max, (k_base * current_x + b_base))
                        coords_top_plus[x] = pretty_line_x(b_base, current_x, current_x)
                        coords_top_plus[y] = current_y
                        _iter_counter += 1

                    # Точка перпендикуляра

                    # Перебор точек по противолинии (-)
                    _iter_counter = 0
                    current_x = x_m
                    current_y = y_m
                    coords_top_minus = {
                        x: current_x,
                        y: current_y,
                    }
                    coords_top_minus.update((f, getattr(current_cloud, f)['M']) for f in not_has_features)
                    while abs(current_cloud.pdf_Rn_dimension(Image(**coords_top_minus))) > current_margin:
                        current_x = current_x - step_len * _iter_counter
                        current_y = pretty_line_y(b_base, y_min, y_max, (k_base * current_x + b_base))
                        coords_top_minus[x] = pretty_line_x(b_base, current_x, current_x)
                        coords_top_minus[y] = current_y
                        _iter_counter += 1

                if not (x, y, cloud_num) in line_equations:
                    b_top_plus = b_frompoint_func(coords_top_plus[x], coords_top_plus[y])
                    b_top_minus = b_frompoint_func(coords_top_minus[x], coords_top_minus[y])

                    _equations = {
                        'func_def': f'''{y}_minus = {k_normal}*x + {b_top_plus};\n{y}_plus = {k_normal}*x + {b_top_minus}''',
                        'k': k_normal,
                        'b_minus': b_top_plus,
                        'b_plus': b_top_minus,
                        f'{y}_plus': lambda xx: pretty_line_y(b_top_plus, y_min, y_max, (k_normal * xx + b_top_plus)),
                        f'{y}_minus': lambda xx: pretty_line_y(b_top_minus, y_min, y_max, (k_normal * xx + b_top_minus)),
                        'b_offset_func': b_frompoint_func,
                    }
                    line_equations[(x, y, cloud_num)] = _equations

                    cloud_plus_points.append((cloud_num, 'plus', Image(**coords_top_plus)))
                    cloud_minus_points.append((cloud_num, 'minus', Image(**coords_top_minus)))

        # Вычислим комбинациями минимальное расстояние
        minlen = None
        cloud_lens = {}
        for points in itertools.permutations(cloud_plus_points + cloud_minus_points, 2):

            # Если это координаты одинаковых облак
            if points[0][0] == points[1][0]:
                continue

            # Если это минусы с минусами
            if points[0][1] == points[1][1]:
                continue

            ml = self.get_between_point_len(points[0][2], points[1][2])
            cloud_lens[str(ml)] = (points[0][2], points[1][2])
            if minlen is None or ml < minlen:
                minlen = ml

        # Координаты середины между пересечениями
        center_coords = {}
        for feature in self.features_names:
            mid_feature = ((getattr(cloud_lens[str(minlen)][1], feature) + getattr(cloud_lens[str(minlen)][0], feature)) / 2)
            center_coords[feature] = mid_feature

        midinm = Image(**center_coords)

        # Вычислим точку перпендикуляра к середине между границами распределения
        endnormal_coords = []
        for feature_r2 in itertools.combinations(self.cloud1.features_names, 2):
            x, y = feature_r2
            x_m = getattr(current_cloud, x)['M']
            y_m = getattr(current_cloud, y)['M']

            x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = self.get_x_y_min_max_and_m(x, y)

            not_has_features = [f for f in current_cloud.features_names if f not in (x, y)]

            _equations1 = line_equations[(x, y, 1)]
            _equations2 = line_equations[(x, y, 2)]

            k_endnormal = (_equations1['k'] + _equations2['k']) / 2
            b_endnormal = (_equations1['b_offset_func'](getattr(midinm, x), getattr(midinm, y)) + _equations2['b_offset_func'](getattr(midinm, x), getattr(midinm, y))) / 2

            if not (x, y, 0) in line_equations:
                _equations = {
                    'func_def': f'''{y} = {k_endnormal}*x + {b_endnormal}''',
                    'k': k_endnormal,
                    'b': b_endnormal,
                    f'{y}': lambda xx: pretty_line_y(b_endnormal, y_min, y_max, (k_endnormal * xx + b_endnormal)),
                    'center_point': midinm,
                }
                line_equations[(x, y, 0)] = _equations

            x1 = pretty_line_x(b_endnormal, getattr(midinm, x), x_min)
            y1 = pretty_line_y(b_endnormal, y_min, y_max, (k_endnormal * x1 + b_endnormal))

            x2 = pretty_line_x(b_endnormal, getattr(midinm, x), x_max)
            y2 = pretty_line_y(b_endnormal, y_min, y_max, (k_endnormal * x2 + b_endnormal))

            endnormal_coords.append({
                x: x1,
                y: y1,
            })
            endnormal_coords.append({
                x: x2,
                y: y2,
            })

        # Вращение прямой относительно точки пересечения
        for feature_r2 in itertools.combinations(self.cloud1.features_names, 2):
            x, y = feature_r2

            k_rotate, b_rotate, left_x, left_y, right_x, right_y = rotate_line_equation(line_equations[(x, y, 0)]['k'],
                                                                                        line_equations[(x, y, 0)]['b'],
                                                                                        (getattr(line_equations[(x, y, 0)]['center_point'], x), getattr(line_equations[(x, y, 0)]['center_point'], y)),  # o_point
                                                                                        (endnormal_coords[0][x], endnormal_coords[0][y]),  # r_point
                                                                                        x_min, x_max, y_min, y_max, step_x
                                                                                        )(fi)
            # Если задано смещение offset_from_O
            if offset_from_O:
                k_rotate, b_rotate, left_x, left_y, right_x, right_y = offset_line_equation_and_points(k_rotate,
                                                                                                       b_rotate,
                                                                                                       (getattr(line_equations[(x, y, 0)]['center_point'], x), getattr(line_equations[(x, y, 0)]['center_point'], y)),  # o_point
                                                                                                       (endnormal_coords[0][x], endnormal_coords[0][y]),  # r_point
                                                                                                       x_min, x_max, y_min, y_max)(offset_from_O)

            line_equations[(x, y, 0)][f'{x}_r_point'] = endnormal_coords[0][x]
            line_equations[(x, y, 0)][f'{y}_r_point'] = endnormal_coords[0][y]

            line_equations[(x, y, 0)][f'{x}_min'] = x_min
            line_equations[(x, y, 0)][f'{x}_max'] = x_max
            line_equations[(x, y, 0)][f'{y}_min'] = y_min
            line_equations[(x, y, 0)][f'{y}_max'] = y_max
            line_equations[(x, y, 0)][f'step_{x}'] = step_x

            line_equations[(x, y, 0)]['k_rotate'] = k_rotate
            line_equations[(x, y, 0)]['b_rotate'] = b_rotate
            line_equations[(x, y, 0)][f'left_{x}'] = left_x
            line_equations[(x, y, 0)][f'left_{y}'] = left_y
            line_equations[(x, y, 0)][f'right_{x}'] = right_x
            line_equations[(x, y, 0)][f'right_{y}'] = right_y

            images.append(Image(**{
                x: left_x,
                y: left_y,
            }))

            images.append(Image(**{
                x: right_x,
                y: right_y,
            }))

        return images, line_equations

    def _test_point_sepline_alpha(self, testpoint: Image, k_sep_line, b_sep_line, left_cloud_num=None):
        """
        С помощьюф функции math.atan2 и уравнению по паралельному переносу и смещению системы координат
        вычислим (угол) квадрант в который попадает линия соединяющая тестируемую точку со смещенным началом координат
        http://www.pm298.ru/dekart.php

        :param testpoint:
        :param k_sep_line:
        :param b_sep_line:
        :param left_cloud_num:
        """
        left_cloud_num_ = left_cloud_num

        ret_alfas = {}

        for feature_r2 in itertools.combinations(self.cloud1.features_names, 2):
            x, y = feature_r2

            x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = self.get_x_y_min_max_and_m(x, y)

            if not left_cloud_num:
                left_cloud_num_ = 1 if x1_m <= x2_m else 2

            not_has_features = [f for f in self.cloud1.features_names if f not in (x, y)]

            # Уравнение линии соединения
            k_base = k((x1_m, y1_m), (x2_m, y2_m))
            b_base = b((x1_m, y1_m), (x2_m, y2_m))

            O_x = pretty_line_x(b_sep_line, (x1_m + x2_m) / 2, (b_sep_line - b_base) / (k_base - k_sep_line))
            O_y = pretty_line_y(b_base, y_min, y_max, (k_base * O_x + b_base))

            test_x, tesy_y = getattr(testpoint, x), getattr(testpoint, y)

            # Угол наклона линии соединения базовой
            alfa_base = math.atan(k_base)

            # Координаты середины отрезка
            mid_point = self.mid_image
            # Координаты точка отрезка соединяющего середину и перпендикуляр
            normal_point = self.get_normal_image_r2_main(x, y)
            k_normal = k((getattr(mid_point, x), getattr(mid_point, y)), (getattr(normal_point, x), getattr(normal_point, y)))
            ### b_normal = b((getattr(mid_point, x), getattr(mid_point, y)), (getattr(normal_point, x), getattr(normal_point, y)))

            # Угол наклона нормали
            alfa_normal = math.atan(k_normal)

            # Угол наклона линии разделения
            alfa_sep_O = math.atan(k_sep_line)

            # Угол смещения
            delta_fi = math.pi - alfa_normal - (math.pi - alfa_sep_O)
            ### delta_fi_grad = math.degrees(math.pi - alfa_normal - (math.pi - alfa_sep_O))

            # Уравнение x_off, y_off для координат относительно линии соединяющей
            def x_off(xx, yy): return ((xx - O_x) * math.cos(alfa_base) + (yy - O_y) * math.sin(alfa_base))

            def y_off(xx, yy): return (-(xx - O_x) * math.sin(alfa_base) + (yy - O_y) * math.cos(alfa_base))

            # Уравнение x_off, y_off для координат относительно линии соединяющей + delta_fi
            def x_off_delta_fi(xx, yy): return ((xx - O_x) * math.cos(alfa_base + delta_fi) + (yy - O_y) * math.sin(alfa_base + delta_fi))

            def y_off_delta_fi(xx, yy): return (-(xx - O_x) * math.sin(alfa_base + delta_fi) + (yy - O_y) * math.cos(alfa_base + delta_fi))

            # Угол смещения относительно перепендикуляра к разделяющей
            ### alfa_off_normal = math.degrees(math.atan2(y_off(test_x, tesy_y), x_off(test_x, tesy_y)))

            # Угол смещения относительно перепендикуляра к разделяющей  + delta_fi
            alfa_off_normal_delta_fi = math.degrees(math.atan2(y_off_delta_fi(test_x, tesy_y), x_off_delta_fi(test_x, tesy_y)))

            if not (x, y) in ret_alfas:
                _equations = {
                    f'{x}_off': x_off,
                    f'{y}_off': y_off,

                    'alfa': alfa_off_normal_delta_fi,
                }
                ret_alfas[(x, y)] = _equations

        return ret_alfas

    def classify_image(self, testpoint: Image, k_sep_line, b_sep_line, left_cloud_num=None):
        """
        Вернуть класс облака которому принадлежит точка

        :param testpoint:
        :param k_sep_line:
        :param b_sep_line:
        :param left_cloud_num:
        """
        left_cloud_num_ = left_cloud_num

        alfas = self._test_point_sepline_alpha(testpoint, k_sep_line=k_sep_line, b_sep_line=b_sep_line, left_cloud_num=left_cloud_num)

        verdicts = []
        for feature_r2 in itertools.combinations(self.cloud1.features_names, 2):
            x, y = feature_r2

            x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = self.get_x_y_min_max_and_m(x, y)

            if not left_cloud_num:
                left_cloud_num_ = 1 if x1_m <= x2_m else 2

            alfa = round(alfas[(x, y)]['alfa'], 2)

            if (180 > alfa > 90) or (-180 < alfa < -90):
                if left_cloud_num_ == 1:
                    verdicts.append(1)
                else:
                    verdicts.append(2)

            elif (90 > alfa > 0) or (-90 < alfa < 0):
                if left_cloud_num_ == 1:
                    verdicts.append(2)
                else:
                    verdicts.append(1)
            else:
                verdicts.append(-1)
                verdicts.append(-1)

        cloud1_count = verdicts.count(1)
        cloud2_count = verdicts.count(2)

        cloud_undefined_count = verdicts.count(-1)

        if cloud1_count > cloud2_count and (not cloud_undefined_count):
            return self.cloud1.klass or 1
        elif cloud1_count < cloud2_count and (not cloud_undefined_count):
            return self.cloud2.klass or 2
        else:
            return None

    def get_optimized_sep_line_points(self, training1: list, training2: list, k_line, b_line, r_point_im: Image, steps=20, left_cloud_num=None):
        """
        Оптимизировать линию, смещая и поворачивая

        :param training1: итерируемое образов для проверки
        :param training2: итерируемое образов для проверки
        :param k_line:
        :param b_line:
        :param r_point_im: точка на прямой не лежащая на линии пересечения ее с соединяющей линией
        :param steps:
        :param left_cloud_num:
        """
        left_cloud_num_ = left_cloud_num

        ret_k = 0
        ret_b = 0
        ret_offset = 0

        images = []
        line_equations = {}

        good_points = defaultdict(dict)
        for feature_r2 in itertools.combinations(self.cloud1.features_names, 2):
            x, y = feature_r2

            x1_m, y1_m, x2_m, y2_m, x_min, x_max, y_min, y_max = self.get_x_y_min_max_and_m(x, y)

            if not left_cloud_num:
                left_cloud_num_ = 1 if x1_m <= x2_m else 2

            # Уравнение линии соединения
            k_base = k((x1_m, y1_m), (x2_m, y2_m))
            b_base = b((x1_m, y1_m), (x2_m, y2_m))

            r_x = getattr(r_point_im, x)
            r_y = getattr(r_point_im, y)

            odd_stnum = 0
            even_stnum = 0
            stvalue = int(abs(getattr(self.cloud1, x)['M'] - getattr(self.cloud2, x)['M']) / 20)
            offsets = []
            for step in range(0, steps + 1):
                iseven = (step % 2 == 0)
                if iseven:
                    even_stnum += 1
                    offsets.append(-even_stnum * stvalue)
                else:
                    odd_stnum += 1
                    offsets.append(odd_stnum * stvalue)

            left_x, left_y, right_x, right_y = x_min, y_min, x_max, y_max

            k_rot_off = k_line
            b_rot_off = b_line

            for offset in offsets:

                O_x = pretty_line_x(b_rot_off, (x1_m + x2_m) / 2, (b_rot_off - b_base) / (k_base - k_rot_off))
                O_y = pretty_line_y(b_base, y_min, y_max, (k_base * O_x + b_base))

                k_rot_off, b_rot_off, left_x, left_y, right_x, right_y = offset_line_equation_and_points(k_rot_off,
                                                                                                         b_rot_off,
                                                                                                         (O_x, O_y),  # o_point
                                                                                                         (r_x, r_y),  # r_point
                                                                                                         x_min, x_max, y_min, y_max
                                                                                                         )(offset)

                k_offset_r = round(k_rot_off, 4)
                b_offset_r = round(b_rot_off, 4)

                if (k_offset_r, b_offset_r, offset) not in good_points[(x, y)]:
                    good_points[(x, y)][(k_offset_r, b_offset_r, offset)] = 0

                for testpoint in itertools.chain(training1, training2):
                    testklass = comparator.classify_image(testpoint, k_sep_line=k_rot_off, b_sep_line=b_rot_off, left_cloud_num=left_cloud_num_)
                    if testklass == testpoint.klass:
                        good_points[(x, y)][(k_offset_r, b_offset_r, offset)] = good_points[(x, y)][(k_offset_r, b_offset_r, offset)] + 1
                    else:
                        #good_points[(x, y)][(k_offset_r, b_offset_r, offset)] = good_points[(x, y)][(k_offset_r, b_offset_r, offset)] - 1
                        pass

                linegp = good_points[(x, y)][(k_offset_r, b_offset_r, offset)]
                # print(f'Линия y = {k_offset_r} * x + {b_offset_r}, Хороших точек: {linegp}')

                for fi_ in range(20, 91, 20):

                    O_x = pretty_line_x(b_rot_off, (x1_m + x2_m) / 2, (b_rot_off - b_base) / (k_base - k_rot_off))
                    O_y = pretty_line_y(b_base, y_min, y_max, (k_base * O_x + b_base))

                    k_rot_off_, b_rot_off_, left_x, left_y, right_x, right_y = rotate_line_equation(k_rot_off,
                                                                                                    b_rot_off,
                                                                                                    (O_x, O_y),  # o_point
                                                                                                    (left_x, left_y),  # r_point
                                                                                                    x_min, x_max, y_min, y_max
                                                                                                    )(fi_)

                    k_rotate_r = round(k_rot_off_, 4)
                    b_rotate_r = round(b_rot_off_, 4)

                    if (k_rotate_r, b_rotate_r, offset) not in good_points[(x, y)]:
                        good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)] = 0

                    for testpoint in itertools.chain(training1, training2):
                        testklass = comparator.classify_image(testpoint, k_sep_line=k_rot_off_, b_sep_line=b_rot_off_, left_cloud_num=left_cloud_num_)
                        if testklass == testpoint.klass:
                            good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)] = good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)] + 1
                        else:
                            #good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)] = good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)] - 1
                            pass

                    linegp = good_points[(x, y)][(k_rotate_r, b_rotate_r, offset)]
                    # print(f'Линия y = {k_rotate_r} * x + {b_rotate_r}, Хороших точек: {linegp}')

            pass

            tmax = max(good_points[(x, y)].values())
            for key, val in good_points[(x, y)].items():
                if val == tmax:
                    ret_k = key[0]
                    ret_b = key[1]
                    ret_offset = key[2]
                    break

            O_x = pretty_line_x(ret_b, (x1_m + x2_m) / 2, (ret_b - b_base) / (k_base - ret_k))
            O_y = pretty_line_y(b_base, y_min, y_max, (k_base * O_x + b_base))

            if not (x, y) in line_equations:
                _equations = {
                    'func_def': f'''{y} = {ret_k}*x + {ret_b}''',
                    'k': ret_k,
                    'b': ret_b,
                    f'{y}': lambda xx: pretty_line_y(ret_b, y_min, y_max, (ret_k * xx + ret_b)),
                    'center_point': Image(**{
                        x: O_x,
                        y: O_y,
                    }),
                }
                line_equations[(x, y)] = _equations

            step_x = abs((x2_m - x1_m) / (steps * 2))

            def formul(xx): return ret_k * xx + ret_b

            if ret_b not in (-inf, inf) and ret_k not in (-inf, inf):

                # "Левая" часть, возрастание и убывание функций

                left_x = x_min
                left_y = formul(left_x)

                if left_y > y_max:
                    while left_y > y_max:
                        left_x += step_x
                        left_y = formul(left_x)

                elif left_y < y_min:
                    while left_y < y_min:
                        left_x += step_x
                        left_y = formul(left_x)

                if left_y < y_min:
                    left_x = x_max
                    left_y = formul(left_x)
                    while left_y < y_min:
                        left_x -= step_x
                        left_y = formul(left_x)

                elif left_y > y_max:
                    left_x = x_max
                    left_y = formul(left_x)
                    while left_y > y_max:
                        left_x += step_x
                        left_y = formul(left_x)

                # "Правая" часть, возрастание и убывание функций

                right_x = x_max
                right_y = formul(right_x)

                if right_y < y_min:
                    while right_y < y_min:
                        right_x -= step_x
                        right_y = formul(right_x)

                elif right_y > y_max:
                    while right_y > y_max:
                        right_x -= step_x
                        right_y = formul(right_x)

                if right_y > y_max:
                    right_x = x_min
                    right_y = formul(right_x)
                    while right_y > y_max:
                        right_x += step_x
                        right_y = formul(right_x)

                elif right_y < y_min:
                    right_x = x_min
                    right_y = formul(right_x)
                    while right_y < y_min:
                        right_x -= step_x
                        right_y = formul(right_x)

                opt_features_x1 = [
                    pretty_line_x(ret_b, O_x, left_x),
                    pretty_line_x(ret_b, O_x, right_x),
                ]

                images.append(Image(**{
                    x: opt_features_x1[0],
                    y: pretty_line_y(ret_b, y_min, y_max, formul(opt_features_x1[0])),
                }))

                images.append(Image(**{
                    x: opt_features_x1[1],
                    y: pretty_line_y(ret_b, y_min, y_max, formul(opt_features_x1[1])),
                }))

            else:
                if ret_k in (-inf, inf) or isnan(ret_k):
                    betta = math.radians(90)
                else:
                    betta = math.atan(ret_k)

                lenline = y_max - O_y

                O_x = O_x + ret_offset

                line_equations[(x, y)][f'center_point'] = Image(**{
                    x: O_x,
                    y: O_y,
                })

                left_x, right_x = lenline * math.cos(betta) + O_x, -lenline * math.cos(betta) + O_x
                left_y, right_y = lenline * math.sin(betta) + O_y, -lenline * math.sin(betta) + O_y

                images.append(Image(**{
                    x: left_x,
                    y: left_y,
                }))

                images.append(Image(**{
                    x: right_x,
                    y: right_y,
                }))

            line_equations[(x, y)][f'{x}_r_point'] = right_x
            line_equations[(x, y)][f'{y}_r_point'] = left_y

            line_equations[(x, y)][f'{x}_min'] = x_min
            line_equations[(x, y)][f'{x}_max'] = x_max
            line_equations[(x, y)][f'{y}_min'] = y_min
            line_equations[(x, y)][f'{y}_max'] = y_max

            line_equations[(x, y)][f'left_{x}'] = left_x
            line_equations[(x, y)][f'left_{y}'] = left_y
            line_equations[(x, y)][f'right_{x}'] = right_x
            line_equations[(x, y)][f'right_{y}'] = right_y

        return images, line_equations

    @staticmethod
    def get_feature_iterator_from_images(images, feature_name):
        for im in images:
            yield getattr(im, feature_name)

    pass


if __name__ == "__main__":

    print('Введите N1: ')
    N1 = input()
    N1 = int(N1) if N1 else 100

    print('Введите N2: ')
    N2 = input()
    N2 = int(N2) if N2 else 100

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
    normal_point = comparator.get_normal_image_r2_main('x', 'y')
    lnorm = mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x", linewidth=0.8, )
    ax.add_line(lnorm)
    # / Координаты точка отрезка соединяющего середину и перпендикуляр

    # ========================
    # Program Body
    # ========================

    # Разделение через минимум Пло́тности вероя́тности

    # Линия перпендикулярная
    sep_points, line_equations = comparator.get_probability_midlane_points(ax=ax, fi=0)
    le = line_equations[('x', 'y', 0)]
    sep_features_x1 = list(CloudComparator.get_feature_iterator_from_images(sep_points, 'x'))
    sep_features_y1 = list(CloudComparator.get_feature_iterator_from_images(sep_points, 'y'))

    ax.add_line(
        mlines.Line2D(
            sep_features_x1,
            sep_features_y1,
            color="black",
            linestyle='dashdot',
            linewidth=1.5,
            marker="")
    )

    # Линия оптимизированная
    opt_sep_points, opt_line_equations = comparator.get_optimized_sep_line_points(cloud1._images[:200], cloud2._images[:200],
                                                                                  le['k'], le['b'],
                                                                                  Image(x=le['x_r_point'], y=le['y_r_point']),
                                                                                  steps=10)
    opt_le = opt_line_equations[('x', 'y')]

    opt_sep_features_x1 = list(CloudComparator.get_feature_iterator_from_images(opt_sep_points, 'x'))
    opt_sep_features_y1 = list(CloudComparator.get_feature_iterator_from_images(opt_sep_points, 'y'))

    ax.add_line(
        mlines.Line2D(
            opt_sep_features_x1,
            opt_sep_features_y1,
            color="red",
            linewidth=2.1,
            marker="")
    )

    # Тестирование точки. Подпись угла
    for testpoint in [random.choice(cloud1._images[:5]),
                      random.choice(cloud2._images[:5]),
                      random.choice(cloud1._images[-5:]),
                      random.choice(cloud2._images[-5:])]:

        testklass = comparator.classify_image(testpoint, k_sep_line=opt_le['k'], b_sep_line=opt_le['b'])
        xtest = round(testpoint.x, 2)
        ytest = round(testpoint.y, 2)
        alfas = comparator._test_point_sepline_alpha(testpoint, k_sep_line=opt_le['k'], b_sep_line=opt_le['b'])
        alfa = round(alfas[('x', 'y')]['alfa'], 2)

        ax.add_line(
            mlines.Line2D(
                [opt_le['center_point'].x, testpoint.x],
                [opt_le['center_point'].y, testpoint.y],
                color="purple",
                linestyle='dotted',
                linewidth=0.8,
                marker="x")
        )
        ax.annotate(f'({xtest},\n{ytest}),\nкласс {testklass}\n угол {alfa}',
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
