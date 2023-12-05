import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def is_empty_string_analysis(string):
    if string.strip() == '':
        return -1
    else:
        return 0


def best_hyperparams_transform(param):
    for key, value in param.items():
        if value % 1 == 0:
            param[key] = int(value)
    return param


def numeric_transform(df, col):
    df[f"{col}"] = pd.to_numeric(df[f"{col}"])


def balance_the_dataset(data, y_name):
    # R = np.random.RandomState(42)

    # Resampling датасета
    yes_index = np.array(data[data[f"{y_name}"] == 1].index)
    no_index = np.array(data[data[f"{y_name}"] == 0].index)

    # Выбор индексов по сумме признаков
    new_no_index = data['sum'].sort_values().iloc[::-1][:len(yes_index)].index
    # new_no_index = R.choice(no_index, len(yes_index))
    # Сгенерируем датасет по индексам
    data = data.loc[list(np.array(new_no_index)) + list(np.array(yes_index))]

    # Перемешаем датасет
    data = shuffle(data, random_state=41)
    data.index = range(len(data))
    return data


def split_to_x_y(df, y_name):
    X = df.iloc[:, df.columns != y_name]
    y = df.iloc[:, df.columns == y_name]

    return X, y
