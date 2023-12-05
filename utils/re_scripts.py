import re


def get_name_and_old(string):
    try:
        s = re.findall(r"\w*.*\s*,\s*\d\d,", string)[0]
    except IndexError:
        pass
    else:
        value = s.split(', ')
        if len(value) > 2:
            pass
        else:
            return value

def get_name(value):
    try:
        name = ''.join(re.findall("[.а-яА-Яa-zA-Z]*", value[0].lower())).strip()
    except:
        pass
    else:
        if name in ['', ' ']:
            pass
        else:
            return name

def get_old(value):
    try:
        old = int(re.findall(r'\d\d', value[1])[0])
    except:
        pass
    else:
        return old


def apply_sum_by_phrase(df, reg_exp, new_col_name, col_for_search, value):
    df[f'{new_col_name}'] = 0
    for reg in reg_exp:
        for i, string in enumerate(df[f'{col_for_search}']):
            try:
                s = re.findall(reg, string)[0]
            except IndexError:
                pass
            else:
                df[f'{new_col_name}'].iloc[i] += value
    return df


def apply_new_col_by_phrase(string, reg_exp, value):
    for reg in reg_exp:
        try:
            s = re.findall(reg, string)[0]
        except IndexError:
            return 0
        else:
            return value


def check_phrase_existence(df, reg_exp, new_col_name, col_for_search, value):
    df[f'{new_col_name}'] = df[f'{col_for_search}'].apply(lambda string: apply_new_col_by_phrase(string, reg_exp, value))
    return df


# Создадим функцию очистки текста
def clean_text(text):
    try:
        text = text.lower()
        text = re.sub(r'[^\sа-яА-Яa-zA-Z0-9@\[\]]', ' ', text)  # Удаляет пунктцацию
        text = re.sub(r'\w*\d+\w*', '', text)  # Удаляет цифры
        text = re.sub('\s{2,}', " ", text)  # Удаляет ненужные пробелы
        text = re.sub('[\n\r]', "", text)  # Удаляет ненужные пробелы
    except AttributeError:
        pass
    else:
        return text


def split_message(string):
    try:
        s = re.findall(r'\w*.*\s*,\s*\d\d\s*,\s*.*\s*–', string)[0]
        text = string.removeprefix(s).lstrip()
    except IndexError:
        pass
    else:
        return text


def split_names(name):
    # print(df['name'])
    anna = re.findall(r'анн\w*|аня|аню\w*|ann\w*|\w*нюта', name)
    marina = re.findall(r'марина|marina', name)
    arina = re.findall(r'ари\w*|\w*рин\w*|ari\w*', name)
    alina = re.findall(r"alin\w*|аля|алин\w*", name)
    alisa = re.findall(r"\w*лис\w*|\w*lis\w*|\w*lic\w*", name)
    albina = re.findall(r'albi\w*|альби\w*', name)
    angelina = re.findall(r"angelin\w*|ангели\w*", name)
    anfisa = re.findall("anfi\w*|anphi\w*|анфи\w*", name)
    alena = re.findall(r"alen\w*|ален\w*|алён\w*", name)
    alesya = re.findall(r"алес\w*|\w*леся\w*|ales\w*|олес\w*|oles\w*", name)
    dasha = re.findall(r"dash\w*|даш\w*|дарь\w*|dar\w*", name)
    diana = re.findall("dian\w*|диан\w*|dina|дин\w*", name)
    evelina = re.findall(r"эля|эве\w*|эль|эли\w*|eveli\w*", name)
    evgenia = re.findall(r"жен\w*|eugen\w*|евг\w*|жек\w*", name)
    eva = re.findall(r"eva|ева", name)
    eleonora = re.findall(r'eleon\w*|элен\w*', name)
    irina = re.findall(r'ир\w*|ir\w*|\w*рэн\w*', name)
    juliya = re.findall(r'jul\w*|юл\w*', name)
    kseniya = re.findall(r'кс\w*|ks\w*', name)
    kristina = re.findall(r'kri\w*|кри\w*', name)
    kamilla = re.findall(r'камил\w*|kamil\w*', name)
    karina = re.findall(r'кари\w*|kari\w*', name)
    katya = re.findall(r'\w*кат\w*|\w*kat\w*|\w*кэт\w*|\w*cat\w*', name)
    larisa = re.findall(r'ларис\w*|laris\w*', name)
    liza = re.findall(r'\w*лиз\w*|\w*liz\w*', name)
    lika = re.findall(r'\w*лик\w*|анж\w*|\w*lika\w*', name)
    lena = re.findall(r'\w*len\w*|elen\w*|\w*лен\w*', name)
    lera = re.findall(r'\w*лер\w*|\w*ler\w*', name)
    masha = re.findall(r'маш\w*|мар\w*|mash\w*|mar\w*', name)
    milena = re.findall(r'mil\w*|мил\w*', name)
    nastya = re.findall(r'\w*нас\w*|\w*nas\w*', name)
    polina = re.findall(r'пол\w*|pol\w*|\w*lina|\w*лин\w*|pali\w*', name)
    sonya = re.findall(r'сон\w*|soph\w*|sof\w*|соф\w*', name)
    sasha = re.findall(r'алекс\w*|саш\w*|sash\w*|\w*aleks\w*|\w*сан\w*', name)
    vladislava = re.findall(r'vlad\w*|влад\w*', name)
    violetta = re.findall(r'вио\w*|vio\w*', name)
    vika = re.findall(r'вик\w*|vik\w*', name)
    dayana = re.findall(r'дая\w*|дарин\w*', name)
    dilyara = re.findall(r'дил\w*|dil\w*', name)
    guzel = re.findall(r'guz\w*|гуз\w*', name)
    galya = re.findall(r'гал\w*|gal\w*', name)
    lubov = re.findall(r'люб\w*|lub\w*|love', name)
    liliya = re.findall(r'лил\w*|lil\w*|\w*leya\w*|лей\w*', name)
    nina = re.findall(r'нин\w*|nin\w*', name)
    nadejda = re.findall(r'над\w*|nad\w*', name)
    natasha = re.findall(r'нат\w*|nat\w*', name)
    oksana = re.findall(r'окс\w*|oks\w*', name)
    olya = re.findall(r'ол\w*|olg\w*|ol\w*', name)
    rita = re.findall(r'рит\w*|rit\w*', name)
    stasya = re.findall(r'\w*ася\w*|\w*asya\w*|stas\w*|стас\w*', name)
    sveta = re.findall(r'свет\w*|svet\w*', name)
    snezha = re.findall(r'снеж\w*|shez\w*', name)
    tanya = re.findall(r'тан\w*|таш\w*|tan\w*|tat\w*|тат\w*', name)
    ulyana = re.findall(r'уль\w*|ul\w*', name)
    valentina = re.findall(r'val\w*|вал\w*', name)
    varya = re.findall(r'вар\w*|var\w*', name)
    vasya = re.findall(r'vas\w*|вас\w*', name)
    veronika = re.findall(r'вер\w*|\w*ник\w*|ver\w*|\w*nik\w*|\w*нич\w*', name)
    yana = re.findall(r'ян\w*|yan\w*', name)
    zarina = re.findall(r'zar\w*|зар\w*', name)

    value = None
    if marina:
        value = 'марина'
    elif alina:
        value = 'алина'
    elif alisa:
        value = 'алиса'
    elif albina:
        value = 'альбина'
    elif angelina:
        value = 'ангелина'
    elif anfisa:
        value = 'анфиса'
    elif alena:
        value = 'алёна'
    elif alesya:
        value = 'алеcя'
    elif dasha:
        value = 'даша'
    elif diana:
        value = 'диана'
    elif evelina:
        value = 'эвелина'
    elif evgenia:
        value = 'женя'
    elif eva:
        value = 'ева'
    elif eleonora:
        value = 'элеонора'
    elif irina:
        value = 'ирина'
    elif juliya:
        value = 'юля'
    elif kseniya:
        value = 'ксюша'
    elif kristina:
        value = 'кристина'
    elif kamilla:
        value = 'камилла'
    elif karina:
        value = 'карина'
    elif katya:
        value = 'катя'
    elif larisa:
        value = 'лариса'
    elif liza:
        value = 'лиза'
    elif lika:
        value = 'анжелика'
    elif lena:
        value = 'лена'
    elif lera:
        value = 'лера'
    elif masha:
        value = 'маша'
    elif milena:
        value = 'милена'
    elif nastya:
        value = 'настя'
    elif polina:
        value = 'полина'
    elif sonya:
        value = 'соня'
    elif sasha:
        value = 'саша'
    elif vladislava:
        value = 'влада'
    elif violetta:
        value = 'виолетта'
    elif vika:
        value = 'вика'
    elif dayana:
        value = 'даяна'
    elif dilyara:
        value = 'диля'
    elif guzel:
        value = 'гузель'
    elif galya:
        value = 'галя'
    elif vika:
        value = 'любовь'
    elif liliya:
        value = 'лиля'
    elif nina:
        value = 'нина'
    elif nadejda:
        value = 'надя'
    elif natasha:
        value = 'наташа'
    elif oksana:
        value = 'надя'
    elif olya:
        value = 'оля'
    elif rita:
        value = 'рита'
    elif stasya:
        value = 'стася'
    elif sveta:
        value = 'света'
    elif snezha:
        value = 'снежа'
    elif tanya:
        value = 'таня'
    elif ulyana:
        value = 'ульяна'
    elif valentina:
        value = 'валя'
    elif varya:
        value = 'варя'
    elif vasya:
        value = 'вася'
    elif veronika:
        value = 'вероника'
    elif yana:
        value = 'яна'
    elif zarina:
        value = 'зарина'
    elif anna:
        value = 'аня'
    elif arina:
        value = 'арина'
    else:
        value = 'unknown'
    return value
