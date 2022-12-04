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
import cmath

np.random.seed(int(time.time()))


def k(p_1, p_2):
    # Уравнение линии (y = kx + b) по двум точкам p_1, p_2

    return (p_2[1] - p_1[1]) / (p_2[0] - p_1[0])


def b(p_1, p_2):
    # Уравнение линии (y = kx + b) по двум точкам p_1, p_2

    return (p_1[1] * p_2[0] - p_1[0] * p_2[1]) / (p_2[0] - p_1[0])


class Image:
    """
    Образ с произвольным набором признаков
    """

    def __init__(self, klass=None, **features):
        """
        :param klass: Класс образа если задан

        features: распаковка набора признаков со значениями, значения должные быть действительными числами
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
    Облако образов в пространстве признаков
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

            features_Ms.append(fsett['M'])

        mean = features_Ms
        cov = self._default_cov

        *features_arrays, = np.random.multivariate_normal(mean, cov, self.size).T

        for features in itertools.zip_longest(*features_arrays):
            ftu = {k: v for k, v in itertools.zip_longest(self.features_names, features)}
            self._images.append(Image(klass=self.klass, **ftu))

    def pdf_Rn_dimension_from_destimator(self, x: Image, kernel_density_estimator):
        """
        Пло́тность вероя́тности (probability density function - PDF) - используя дестиматор

        scipy.stats.gaussian_kde
        KernelDensity

        https://scikit-learn.org/stable/modules/density.html
        https://stackoverflow.com/questions/52160088/python-fast-kernel-density-estimation-probability-density-function

        from fastkde.fastKDE import pdf

        def get_pdf(data):
            y, x = pdf(data)
            return x, y

        :param x:
        """
        if x.dimensionality != self.dimensionality:
            raise ValueError("Размерность образа и облака не соотносятся")

        return None

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

    @property
    def mid_image(self) -> Image:
        """
        Точка на середине отрезка соединяющего облака

        """
        coords = {}
        for feature in self.features_names:
            mid_feature = ((getattr(self.cloud2, feature)['M'] + getattr(self.cloud1, feature)['M']) / 2)
            coords[feature] = mid_feature

        return Image(**coords)

    @property
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
        Координаты нормали к отрезку соединяющему облака

        :param feature_name1: какой признак взять за x
        :param feature_name2: какой признак взять за y
        :param znak:
        """
        mid_image = self.mid_image

        # Обрежем размерность до R2
        fe_r2 = [feature_name1, feature_name2]

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

        # Заполним координаты недолстающих признаков нулями
        not_has_features = [x for x in self.features_names if x not in fe_r2]
        coords.update((k, 0) for k in not_has_features)

        return Image(**coords)

    def get_probability_circle_points(self, cloud_num=1, margin_of_error=0.0005, ax=None):
        """
        Нарисовать линию (окружность) разделения между облаками

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

    def get_probability_midlane_points(self, margin_of_error=0.0005, ax=None):
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

        for cloud_num in (1, 2):
            current_cloud = self.cloud1 if cloud_num == 1 else self.cloud2
            current_margin = margin_of_error1 if cloud_num == 1 else margin_of_error2
            current_step_accuracy = step_accuracy1 if cloud_num == 1 else step_accuracy2

            for feature_r2 in itertools.combinations(current_cloud.features_names, 2):
                x, y = feature_r2
                x_m = getattr(current_cloud, x)['M']
                y_m = getattr(current_cloud, y)['M']

                x1_m = getattr(self.cloud1, x)['M']
                y1_m = getattr(self.cloud1, y)['M']
                x2_m = getattr(self.cloud2, x)['M']
                y2_m = getattr(self.cloud2, y)['M']

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
                    current_y = k_base * current_x + b_base
                    coords_top_plus[x] = current_x
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
                    current_y = k_base * current_x + b_base
                    coords_top_minus[x] = current_x
                    coords_top_minus[y] = current_y
                    _iter_counter += 1

                if not (x, y, cloud_num) in line_equations:
                    b_top_plus = b_frompoint_func(coords_top_plus[x], coords_top_plus[y])
                    b_top_minus = b_frompoint_func(coords_top_minus[x], coords_top_minus[y])

                    _equations = {
                        'func_def': f'''{y} = {k_normal}*x + {b_top_plus};\n{y} = {k_normal}*x + {b_top_minus}''',
                        'k': k_normal,
                        'b_minus': b_top_plus,
                        'b_plus': b_top_minus,
                        f'{y}_plus': lambda xx: k_normal * xx + b_top_plus,
                        f'{y}_minus': lambda xx: k_normal * xx + b_top_minus,
                        'b_offset_func': b_frompoint_func,
                    }
                    line_equations[(x, y, cloud_num)] = _equations

                    cloud_plus_points.append((cloud_num, Image(**coords_top_plus)))
                    cloud_minus_points.append((cloud_num, Image(**coords_top_minus)))

        # Вычислим комбинациями минимальное расстояние
        minlen = max(M1max, M2max)
        cloud_lens = {}
        for points in itertools.permutations(cloud_plus_points + cloud_minus_points, 2):
            if points[0][0] == points[1][0]:
                continue
            ml = self.get_between_point_len(points[0][1], points[1][1])
            cloud_lens[str(ml)] = (points[0][1], points[1][1])
            if ml < minlen:
                minlen = ml

        # Координаты середины между пересечениями
        center_coords = {}
        for feature in self.features_names:
            mid_feature = ((getattr(cloud_lens[str(minlen)][1], feature) + getattr(cloud_lens[str(minlen)][0], feature)) / 2)
            center_coords[feature] = mid_feature

        images.append(Image(**center_coords))

        # Вычислим точку перпендикуляра к середине
        endnormal_coords = []
        for feature_r2 in itertools.combinations(current_cloud.features_names, 2):
            x, y = feature_r2
            x_m = getattr(current_cloud, x)['M']
            y_m = getattr(current_cloud, y)['M']

            x1_m = getattr(self.cloud1, x)['M']
            y1_m = getattr(self.cloud1, y)['M']
            x2_m = getattr(self.cloud2, x)['M']
            y2_m = getattr(self.cloud2, y)['M']

            not_has_features = [f for f in current_cloud.features_names if f not in (x, y)]

            _equations = line_equations[(x, y, 1)]

            k_endnormal = _equations['k']
            b_endnormal = _equations['b_offset_func'](getattr(images[-1], x), getattr(images[-1], y))

            # Вычислим минимальные и максимальные x
            x_min = min(itertools.chain(self.cloud1.get_feature_iterator(x), self.cloud2.get_feature_iterator(x)))
            x_max = max(itertools.chain(self.cloud1.get_feature_iterator(x), self.cloud2.get_feature_iterator(x)))

            x1 = x_min
            y1 = k_endnormal * x1 + b_endnormal
            endnormal_coords.append({
                x: x1,
                y: y1,
            })

            x2 = x_max
            y2 = k_endnormal * x2 + b_endnormal
            endnormal_coords.append({
                x: x2,
                y: y2,
            })

        images.append(Image(**endnormal_coords[0]))
        images.append(Image(**endnormal_coords[1]))

        return images, line_equations

    @staticmethod
    def get_feature_iterator_from_images(images, feature_name):
        for im in images:
            yield getattr(im, feature_name)

    pass


if __name__ == "__main__":
    cloud1 = ClassNormalCloud(100, x={'M': 200, 'D': 1000}, y={'M': 500, 'D': 800}, klass=1)
    cloud1.fill_cloud_Rn_dimension()

    cloud2 = ClassNormalCloud(100, x={'M': 700, 'D': 8000}, y={'M': 700, 'D': 1000}, klass=2)
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
    normal_point = comparator.get_normal_image_r2_main('x', 'y')
    lnorm = mlines.Line2D([mid_point.x, normal_point.x], [mid_point.y, normal_point.y], color="green", linestyle="-", marker="x")
    ax.add_line(lnorm)
    # / Координаты точка отрезка соединяющего середину и перпендикуляр

    # ========================
    # Program Body
    # ========================

    # Разделение через минимум Пло́тности вероя́тности
    sep_points, line_equations = comparator.get_probability_midlane_points(ax=ax)
    sep_features_x1 = list(CloudComparator.get_feature_iterator_from_images(sep_points, 'x'))
    sep_features_y1 = list(CloudComparator.get_feature_iterator_from_images(sep_points, 'y'))

    ax.scatter(sep_features_x1, sep_features_y1, color="#FDB94D", s=300)

    # Разделение через минимум Пло́тности вероя́тности
    sep_plane_images, circle_equations1 = comparator.get_probability_circle_points(1, ax=ax)

    sep_features_x1 = list(itertools.chain(CloudComparator.get_feature_iterator_from_images(sep_plane_images, 'x')))
    sep_features_y1 = list(itertools.chain(CloudComparator.get_feature_iterator_from_images(sep_plane_images, 'y')))
    for i in range(1, len(sep_plane_images), 1):
        ax.add_line(
            mlines.Line2D(
                [sep_plane_images[i - 1].x, sep_plane_images[i].x],
                [sep_plane_images[i - 1].y, sep_plane_images[i].y],
                color="#e6188c",
                marker="x")
        )

    sep_plane_images, circle_equations2 = comparator.get_probability_circle_points(2, ax=ax)

    sep_features_x1 = list(itertools.chain(CloudComparator.get_feature_iterator_from_images(sep_plane_images, 'x')))
    sep_features_y1 = list(itertools.chain(CloudComparator.get_feature_iterator_from_images(sep_plane_images, 'y')))
    for i in range(1, len(sep_plane_images), 1):
        ax.add_line(
            mlines.Line2D(
                [sep_plane_images[i - 1].x, sep_plane_images[i].x],
                [sep_plane_images[i - 1].y, sep_plane_images[i].y],
                color="#44ec86",
                marker="x")
        )

    # ========================
    # / Program Body
    # ========================

    plt.show()
    sys.exit()
