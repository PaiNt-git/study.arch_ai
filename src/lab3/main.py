import sys
import numpy as np


from collections import defaultdict
import itertools
import re


def notempty(x): return bool(x and not x.isspace())


def lchunks(lst, n):
    for i_ in range(0, len(lst), n):
        yield lst[i_:i_ + n]


class Classificator:
    """
    Классификатор блоков пикселей (символов), записанных в текстовых файлах
    При определении класс подгружает эталоны и определеяте производные значения

    """

    # Init Class

    def __new__(cls, *args, **kwargs):
        if cls is Classificator:
            raise TypeError()
        return super().__new__(cls)

    ETALON_PIXEL_DATA = defaultdict(list)
    MAX_HAMM_DIST = 0

    MATRIX_MAX_X = 0
    MATRIX_MAX_Y = 0

    def __pixels_load(cls, filepath, moduletdict):

        with open(filepath, 'r') as fl:
            counter = 0
            for line in fl.readlines():
                if counter > 10:
                    fl.close()
                    return
                if notempty(line):
                    line = line.strip()
                    moduletdict[counter].append(line)
                else:
                    counter += 1

    __pixels_load(None, 'etalon.txt', ETALON_PIXEL_DATA)

    if len(ETALON_PIXEL_DATA):
        MATRIX_MAX_X = len(ETALON_PIXEL_DATA[0][0])
        MATRIX_MAX_Y = len(ETALON_PIXEL_DATA[0])

        __rl2 = MATRIX_MAX_Y // 2
        MAX_HAMM_DIST = __rl2 * (MATRIX_MAX_X // 2)  # Считаем что макс растояние Хемминга для одной цифры должно быть не более четверти пикселей

    pixels_load = classmethod(__pixels_load)

    # Other methods

    @staticmethod
    def quartering(digitsimb: list):
        """
        Поделитть пиксели на четверти

        :param digitsimb: список вида:
            [
            '*******',
            '*....*',
            '*....*',
            '*....*',
            '*....*',
            '*....*',
            '*******',
            ]
        """
        x_len = 0
        y_len = len(digitsimb)
        if y_len and notempty(digitsimb[0]):
            x_len = len(digitsimb[0])

        x_count = x_len // 2
        y_count = y_len // 2

        devided_x = []

        for i in range(y_len):
            cnks = list(lchunks(digitsimb[i], x_count))
            cnks = [cnks[0], ''.join(cnks[1:])]
            devided_x.append(cnks)

        devided_y = list(lchunks(devided_x, y_count))
        devided_y = [devided_y[0], list(itertools.chain.from_iterable(devided_y[1:]))]

        quadrants = []

        for i, j in itertools.product(range(2), range(2)):
            quadrants.append([x[j] for x in devided_y[i]])

        return quadrants

    @staticmethod
    def bin_vectorize_block(digitblock: list):
        """
        С помощью numpy предстваить список "пикселей" как бинарный вектор

        :param digitblock: список строк пикселей типаЖ
            [
            '****',
            '*..',
            '*..',
            ]
        """
        return np.array(list(map(lambda x: 1 if x == '*' else 0, itertools.chain.from_iterable(digitblock))))

    @staticmethod
    def vect_dist(vect1: np.ndarray, vect2: np.ndarray):
        """
        Математическое растояние между векторами, (не пригодилось в задаче)

        :param vect1:
        :param vect2:
        """
        return np.linalg.norm(vect1 - vect2)

    @staticmethod
    def hamm_dist(etalonvect: np.ndarray, vect: np.ndarray):
        """
        Вычисление расстояния хемминга между бинарными векторми с помощью numpy

        :param etalonvect:
        :param vect:
        """
        vect_dif = etalonvect - vect  # Разность векторов. Если совпадает координата - то на этой позиции будет 0
        nz = np.nonzero(vect_dif)  # Выделить из вектора разности ненулевые значения и вернуть позции на которых они есть
        sp = np.shape(nz[0])  # Размерность массива + длинна, в случае вектора кортеж из одного элемента
        return sp[0]

    @staticmethod
    def get_quadrant_koef_for_symbol(symbol: str, quadrant: int):
        """
        Коэффициент (вес) четверти цифры, символа, (не пригодилось в задаче)

        :param symbol:
        :param quadrant:
        """
        return 1

    @staticmethod
    def get_generalized_hamm_dist(symbol: str, quad_0_dist, quad_1_dist, quad_2_dist, quad_3_dist):
        """
        Общее расстояние Хемминга за символ (по четвертям сумма)

        :param symbol:
        :param quad_0_dist:
        :param quad_1_dist:
        :param quad_2_dist:
        :param quad_3_dist:
        """
        qk = Classificator.get_quadrant_koef_for_symbol
        koefs = [
            qk(symbol, 0),
            qk(symbol, 1),
            qk(symbol, 2),
            qk(symbol, 3),
        ]
        return (koefs[0] * quad_0_dist + koefs[1] * quad_1_dist + koefs[2] * quad_2_dist + koefs[3] * quad_3_dist) / sum(koefs)


class PixelDigit(Classificator):
    """
    Субкласс Класификатора символов - цифер
    """

    def __init__(self, string_with_lb: str):
        """
        Инициализация с текстом с обрывами строк в котором записаны пиксели

        :param string_with_lb:
        """
        self.digit = -1
        self.exact = False
        self.recog_hamm_dist = -1
        self.simple_hamm_dist = -1
        self._pixel_data = list(map(lambda x: x.strip(), string_with_lb.splitlines()))
        self.percent_like = 0

        self.recognite()

    def recognite(self):
        """
        Распознать
        """
        digit_hamings = []
        digit_simple_hammings = []
        for i in range(10):
            etalon_quads = self.quartering(self.ETALON_PIXEL_DATA[i])
            reco_quads = self.quartering(self._pixel_data)

            vb = self.bin_vectorize_block

            digit_simple_hammings.append(self.hamm_dist(vb(self.ETALON_PIXEL_DATA[i]), vb(self._pixel_data)))

            hammings = [
                self.hamm_dist(vb(etalon_quads[0]), vb(reco_quads[0])),
                self.hamm_dist(vb(etalon_quads[1]), vb(reco_quads[1])),
                self.hamm_dist(vb(etalon_quads[2]), vb(reco_quads[2])),
                self.hamm_dist(vb(etalon_quads[3]), vb(reco_quads[3])),
            ]

            genham = self.get_generalized_hamm_dist(i, *hammings)
            if genham == 0 == hammings[0] == hammings[1] == hammings[2] == hammings[3]:
                self.recog_hamm_dist = 0
                self.exact = True
                self.digit = i
                self.simple_hamm_dist = digit_simple_hammings[i]
                self.percent_like = 100
                return

            digit_hamings.append(genham)

        self.exact = False
        self.recog_hamm_dist = min(digit_hamings)
        self.digit = digit_hamings.index(self.recog_hamm_dist)
        self.simple_hamm_dist = digit_simple_hammings[self.digit]

        if self.simple_hamm_dist > self.MAX_HAMM_DIST:
            self.digit = -1

        self.percent_like = (
            ((self.MATRIX_MAX_X * self.MATRIX_MAX_Y) -
             self.simple_hamm_dist) * 100 /
            (self.MATRIX_MAX_X * self.MATRIX_MAX_Y)
        ) if self.simple_hamm_dist > 0 else 100

    def __eq__(self, other):
        """
        Перегрузка операции равенства
        :param other:
        """
        return True if (self - other) < self.MAX_HAMM_DIST else False

    def __sub__(self, other):
        """
        Перегрузка операции разности (отличия) - расстояние Хемминга между цифрами
        :param other:
        """
        etalonvect = self.bin_vectorize_block(self.ETALON_PIXEL_DATA[self.digit])
        othervect = self.bin_vectorize_block(self._pixel_data)
        return self.hamm_dist(etalonvect, othervect)


class Clock:
    """
    Класс Показаний часов
    """

    def devide_digits(self, string_time: str):
        lipixels = string_time.splitlines(False)

        maxx = Classificator.MATRIX_MAX_X
        maxy = Classificator.MATRIX_MAX_Y

        time_digits = defaultdict(str)

        for i in range(len(lipixels)):
            lipixels[i] = re.sub('[^.*]+', '', lipixels[i])

            cnks = list(lchunks(lipixels[i], maxx))

            time_digits[0] = '\n'.join([time_digits[0], cnks[0]]).strip()
            time_digits[1] = '\n'.join([time_digits[1], cnks[1]]).strip()
            time_digits[2] = '\n'.join([time_digits[2], cnks[2]]).strip()
            time_digits[3] = '\n'.join([time_digits[3], cnks[3]]).strip()

        return time_digits[0], time_digits[1], time_digits[2], time_digits[3]

    def __init__(self, string_time: str):
        """
        Передаем значение показаний записанное с пробельными разделитлями, 4 цифры, по типу:

        ******      ******              ******      ******
        *....*      *....*              *....*      *....*
        .....*      *....*              *....*      *....*
        *....*      *....*      +       *....*      *.....
        *....*      *....*              *....*      *....*
        *....*      *....*              *....*      *....*
        *....*      *....*      +       .....*      *....*
        *....*      *....*              *....*      *....*
        ******      ******              ******      ******

        :param string_time:
        """
        self._d1, self._d2, self._d3, self._d4 = self.devide_digits(string_time)

        self.first_digit = PixelDigit(self._d1)
        self.first_digit.digit_pre = self.first_digit.digit

        self.second_digit = PixelDigit(self._d2)
        self.second_digit.digit_pre = self.second_digit.digit

        self.third_digit = PixelDigit(self._d3)
        self.third_digit.digit_pre = self.third_digit.digit

        self.four_digit = PixelDigit(self._d4)
        self.four_digit.digit_pre = self.four_digit.digit

        # Поправки для контекста "Часы и минуты"
        if self.first_digit.digit > 2:
            self.first_digit.digit = -1
        else:
            if self.first_digit.digit == 2 and self.second_digit.digit > 4:
                self.second_digit.digit = -1

        if self.third_digit.digit > 5:
            self.third_digit.digit = -1

        if self.first_digit.exact == self.second_digit.exact == self.third_digit.exact == self.four_digit.exact:
            self.exact = self.first_digit.exact
        else:
            self.exact = False

        # Значения хемминга для показаний времени
        self.simple_hamm_dist = self.first_digit.simple_hamm_dist + self.second_digit.simple_hamm_dist + self.third_digit.simple_hamm_dist + self.four_digit.simple_hamm_dist

        self.percent_like = (
            (((Classificator.MATRIX_MAX_X * Classificator.MATRIX_MAX_Y) * 4) -
             self.simple_hamm_dist) * 100 /
            ((Classificator.MATRIX_MAX_X * Classificator.MATRIX_MAX_Y) * 4)
        ) if self.simple_hamm_dist > 0 else 100

    def __str__(self):
        if self.first_digit.digit != -1 and self.second_digit.digit != -1 and self.third_digit.digit != -1 and self.four_digit.digit != -1:
            return f'{self.first_digit.digit}{self.second_digit.digit}:{self.third_digit.digit}{self.four_digit.digit}'
        return '[Невозможно определить]'

    time = property(__str__)


if __name__ == "__main__":

    print('Введите строку для распознавания: ')
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not notempty(line):
            break
        contents.append(line)

    txt = '\n'.join(contents)

    rec_time = Clock(txt)

    print(f'''\n
Введенное значение:
{txt}

Предполагаемое время:
{rec_time.time}

Точное совпадение:
{rec_time.exact}

Расстояние Хемминга (общее):
{rec_time.simple_hamm_dist}
Процент похожести (общий):
{rec_time.percent_like}


Процент похожести (десятки часов, цифра:{rec_time.first_digit.digit_pre}):
{rec_time.first_digit.percent_like}
Процент похожести (часы, цифра:{rec_time.second_digit.digit_pre}):
{rec_time.second_digit.percent_like}
Процент похожести (десятки минут, цифра:{rec_time.third_digit.digit_pre}):
{rec_time.third_digit.percent_like}
Процент похожести (минуты, цифра:{rec_time.four_digit.digit_pre}):
{rec_time.four_digit.percent_like}

    ''')
