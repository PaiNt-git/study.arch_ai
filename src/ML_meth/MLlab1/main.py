import numpy as np
import matplotlib.pyplot as plt

#cell 1
if __name__ == "__main__":
    points = {
        'green': [[1, 7], [7, 3]],
        'blue': [[4, 2], [9, 1]]
    }

    count_of_points = 0  # размер обучающей выборки
    for point_class in points:
        count_of_points += len(points[point_class])

    w = [0, -1]  # нач. значение вектора w

    def a(x): return np.sign(x[0] * w[0] + x[1] * w[1])  # решающее правило
    N = 50   # максимальное число итераций
    L = 0.1  # шаг изменения веса
    e = -0.05  # небольшая добавка w0 для зазора м/ду раздел. линией и классами
    last_err_index = None  # индекс последнего ошибоч. наблюдения

    for n in range(N):
        count_of_right_answer = 0
        for point_class in points:
            multiplier = 1
            if point_class == 'green':
                multiplier = -1
            for point in points[point_class]:
                if multiplier * a(point) < 0:
                    w[0] = w[0] + L * multiplier
                    last_err_index = multiplier
                else:
                    count_of_right_answer += 1
        if count_of_right_answer == count_of_points:
            print('break')
            break

    w_0 = 0  # смещение
    if last_err_index:
        w[0] += e * last_err_index + w_0
    print_w = f"w = transpose([w_0, w_1, w_2]) = transpose([{w_0}, {w[0]}, {w[1]}])"

    # определяем координаты разделяющей линии
    line_x = [xy[0] for point_class in points for xy in points[point_class]]
    line_x_coords = min(line_x), max(line_x)
    line_y_coords = w[0] * line_x_coords[0], w[0] * line_x_coords[1]

    # вычисляем отступы
    point_margin = {
        'green': [],
        'blue': []
    }
    for point_class in points:
        multiplier = 1
        if point_class == 'green':
            multiplier = -1
        for point in points[point_class]:
            point_margin[point_class].append(
                round(multiplier * (w[0] * point[0] + w[1] * point[1]), 2)
            )

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    ax.set_title(print_w, color='black')
    ax.plot(line_x_coords, line_y_coords, color='red',
            label=f'Разделяющая линия, {w[0]} * x + {w[1]} * y + {w_0} = 0'
            )

    for point_class in points:
        ax.scatter(
            list(map(lambda x: x[0], points[point_class])),
            list(map(lambda x: x[1], points[point_class])),
            color=point_class,
            label=f"Образы {point_class}, Отступы: {point_margin[point_class]}"
        )
    ax.tick_params(labelcolor='indigo')
    ax.legend()
    ax.grid()
    plt.show()


#cell 2


#cell 3


if __name__ == "__main__":
    pass
