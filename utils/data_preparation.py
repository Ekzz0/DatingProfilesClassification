from typing import Tuple, List

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from .data_processing import numeric_transform, is_empty_string_analysis
from .re_scripts import split_names, split_message, clean_text, check_phrase_existence, apply_sum_by_phrase, \
    get_name_and_old, get_name, get_old


def select_name_and_old(df: pd.DataFrame):
    df['name_old'] = None
    df['name_old'] = df['message'].apply(get_name_and_old)

    df['name'] = None
    df['old'] = None
    df['name'] = df['name_old'].apply(get_name)
    df['old'] = df['name_old'].apply(get_old)
    df.drop(columns='name_old', inplace=True)
    df.dropna(subset=['name', 'old'], inplace=True)
    df.index = range(len(df))

    # Получим уникальные имена
    df['name'] = df['name'].apply(split_names)

    return df


def select_the_main_features(path: str):
    pd.options.mode.chained_assignment = None  # default='warn'
    df = pd.read_csv(path)
    # df = pd.read_csv("../data/to_test.csv")
    df.drop(columns='Unnamed: 0', inplace=True)

    # Замена сердечка и лайка на 1 и 0 соответственно
    oEnc = OrdinalEncoder(categories=[['👎', '❤️']])
    df['class'] = oEnc.fit_transform(df['class'].values.reshape(-1, 1))

    # Разделим сообщение на имя и возраст
    df = select_name_and_old(df)

    # Разделим имена на: крутые/нормальные/плохие/не указанные
    best_names = ['ксюша', 'вика', 'даша', 'аня', 'ирина', 'маша', 'кристина', 'алиса', 'лиза', 'ева', 'саша', 'юля',
                  'оля', 'диана', 'лера', 'женя', 'алёна',
                  'анфиса', 'лариса', 'карина', 'полина', 'эвелина', 'таня']
    cool_names = ['соня', 'арина', 'влада', 'надя', 'катя', 'вероника', 'вероника', 'анжелика', 'лена', 'наташа',
                  'стася', 'яна',
                  'марина', 'ульяна', 'виолетта', 'гузель', 'света', 'альбина', 'варя', 'камилла', 'милена', 'диля',
                  'лиля', 'элеонора', 'алеcя', 'рита', 'вася',
                  'зарина', 'валя', 'галя', 'нина', 'ангелина']
    bad_names = ['даяна', 'алина', 'настя', 'снежа']
    unknown = ['unknown']

    df['name'].mask(df.name.isin(best_names), 'best_name', inplace=True)
    df['name'].mask(df.name.isin(cool_names), 'norm_name', inplace=True)
    df['name'].mask(df.name.isin(bad_names), 'bad_name', inplace=True)
    df['name'].mask(df.name.isin(unknown), 'unknown_name', inplace=True)

    # 3 - best_name
    # 2 - norm_name
    # 1 - bad_name
    # 0 - unknown_name
    oEnc = OrdinalEncoder(categories=[['unknown_name', 'bad_name', 'norm_name', 'best_name']])
    df['name'] = oEnc.fit_transform(df['name'].values.reshape(-1, 1))

    # Данные для дальнейшей проверки "флагов"
    cols = []
    col_for_search = 'message'

    # Проверка, указано ли расстояние
    reg_exp = [r"\d*\sкм\b|\d*\sм\b"]
    new_col_name = 'distance_to'
    cols.append(new_col_name)
    value = 1
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указан ли рост
    reg_exp = [r"рост\s*\d\d\d"]
    new_col_name = 'height'
    cols.append(new_col_name)
    value = 1
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указано ли 'ищу парня итд...'
    reg_exp = [r"ищу\sотношени.|ищу\sпарня|серь.зные|мужчин|в\sперспективе"]
    new_col_name = 'romantic_relationships'
    cols.append(new_col_name)
    value = 2
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указано ли 'fwb|ons...'
    reg_exp = [r"fwb|ons|онс|фвб|аббревиатур"]
    new_col_name = 'fwb_ons'
    cols.append(new_col_name)
    value = 1
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указано ли, что человек ничего не ищет
    reg_exp = [r"ищу\s*общение|друга|друзей|Простого\sчеловеческого"]
    new_col_name = 'not_looking_for'
    cols.append(new_col_name)
    value = -2
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указано ли, что человек ничего не ищет
    reg_exp = [r"инст|inst"]
    new_col_name = 'inst'
    cols.append(new_col_name)
    value = 1
    df = check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указаны ли отпугивающие фразы
    reg_exp = [r"там\sпосмотрим",
               r"цветы",
               r"тату|пиво|пить|бар|пью",
               r"size|не\sхудышк|в\sтеле",
               r"реб.н|развед.нка|с\sприцепом",
               r"курю|сигарет|матерюсь",
               r"есть\sпарень|замужем|жената|не\sищу\sотношени\w|парня\sне\sищу|не\sищу\sпарня|состою\sв\sотнош"]
    new_col_name = 'cringe_phrase'
    cols.append(new_col_name)
    value = -3
    df = apply_sum_by_phrase(df, reg_exp, new_col_name, col_for_search, value)

    # Проверка, указаны ли совпадающие интересы
    reg_exp = [r"учусь|магистратур|вуз|колледж|учебе",
               r"работа",
               r"игр|фильм|сериал|кино|читаю|кофе|музык",
               r"политех",
               r"лыж|коньк|сноуборд",
               r"программир|физик|математик",
               r"рок|аниме|фотограф|гитар|спорт",
               r"гулять|прогулк|сходить|живые\sвстречи|встречи",
               r"готовить|вкусно",
               r"вместе"]
    new_col_name = 'similar_interests'
    cols.append(new_col_name)
    value = 3
    df = apply_sum_by_phrase(df, reg_exp, new_col_name, col_for_search, value)

    # Выделим текст сообщения (не включая имя, возраст)
    df['text'] = ''
    df['text'] = df['message'].apply(split_message)

    # Очистка сообщения от ненужных символов
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_text'] = df['clean_text'].fillna('')
    df.drop(columns='text', inplace=True)

    # Проверка, пустая ли анкета
    df['is_empty'] = 0
    cols.append('is_empty')
    df['is_empty'] = df['clean_text'].apply(is_empty_string_analysis)

    # Длина сообщения в анкете
    df['message_len'] = 0
    df['message_len'] = df['clean_text'].apply(len)
    df['sum'] = df[cols].sum(axis=1)
    # Перевод всех признаков в числовой формат
    for col in cols:
        numeric_transform(df, col)

    return df, cols
