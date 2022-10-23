import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_random_dataset(size):
    """Генерация случайного набора данных, которая следует за квадратичным распределением
    """
    x = []
    y = []
    target = []
    for i in range(size):
        # класс 0
        x.append(np.round(random.uniform(0, 2.5), 1))
        y.append(np.round(random.uniform(0, 20), 1))
        target.append(0)
        # класс 1
        x.append(np.round(random.uniform(1, 5), 2))
        y.append(np.round(random.uniform(20, 25), 2))
        target.append(1)
        x.append(np.round(random.uniform(3, 5), 2))
        y.append(np.round(random.uniform(5, 25), 2))
        target.append(1)
    df_x = pd.DataFrame(data=x)
    df_y = pd.DataFrame(data=y)
    df_target = pd.DataFrame(data=target)
    data_frame = pd.concat([df_x, df_y], ignore_index=True, axis=1)
    data_frame = pd.concat([data_frame, df_target], ignore_index=True, axis=1)

    data_frame.columns = ['x', 'y', 'target']
    return data_frame


# Генерация набора данных
size = 100
dataset = generate_random_dataset(size)
features = dataset[['x', 'y']]
label = dataset['target']
# Получение 20% от набора данных для обучения
test_size = int(np.round(size * 0.2, 0))
# Разделение набора данных на обучающие и тестовые наборы
x_train = features[:-test_size].values
y_train = label[:-test_size].values
x_test = features[-test_size:].values
y_test = label[-test_size:].values
# Построение графика для обучающего набора
fig, ax = plt.subplots(figsize=(12, 7))
# Удаление верхней и правой границ
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
# Добавление основных линий сетки
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
ax.scatter(features[:-test_size]['x'], features[:-test_size]['y'], color="#8C7298")
plt.show()
