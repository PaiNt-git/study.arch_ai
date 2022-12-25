import sys
import numpy as np


from collections import defaultdict
import itertools
import re


def notempty(x): return bool(x and not x.isspace())


def lchunks(lst, n):
    for i_ in range(0, len(lst), n):
        yield lst[i_:i_ + n]


class NNPixelDigit():
    """
    Класс цифры
    """

    def __init__(self, string_with_lb: str):
        """
        Инициализация с текстом с обрывами строк в котором записаны пиксели

        :param string_with_lb:
        """
        self.digit = -1
        self._pixel_data = list(map(lambda x: x.strip(), string_with_lb.splitlines()))


class Clock:
    """
    Класс Показаний часов
    """

    ETALON_PIXEL_DATA = defaultdict(list)

    MATRIX_MAX_X = 0
    MATRIX_MAX_Y = 0

    def devide_digits(self, string_time: str):
        lipixels = string_time.splitlines(False)

        maxx = self.MATRIX_MAX_X
        maxy = self.MATRIX_MAX_Y

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
        def pixels_load(filepath, moduletdict):
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

        pixels_load('etalon.txt', self.ETALON_PIXEL_DATA)

        if len(self.ETALON_PIXEL_DATA):
            MATRIX_MAX_X = len(self.ETALON_PIXEL_DATA[0][0])
            MATRIX_MAX_Y = len(self.ETALON_PIXEL_DATA[0])

        self._d1, self._d2, self._d3, self._d4 = self.devide_digits(string_time)

        self.first_digit = NNPixelDigit(self._d1)

        self.second_digit = NNPixelDigit(self._d2)

        self.third_digit = NNPixelDigit(self._d3)

        self.four_digit = NNPixelDigit(self._d4)

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
