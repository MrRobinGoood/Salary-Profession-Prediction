import pandas as pd
from html.parser import HTMLParser
import re
import numpy as np
import math
df = pd.read_csv('vprod_train/TRAIN_SAL.csv', encoding='utf-8')
clear_df = df[['id','academic_degree','accommodation_capability', 'additional_requirements','busy_type','career_perspective', 'education', 'education_speciality', 'is_mobility_program', 'need_medcard', 'other_vacancy_benefit', 'position_requirements', 'position_responsibilities', 'regionName', 'regionNameTerm', 'company_business_size', 'required_certificates', 'required_drive_license', 'required_experience', 'salary', 'salary_min', 'salary_max', 'schedule_type', 'professionalSphereName', 'languageKnowledge', 'hardSkills', 'softSkills']]

for_clear = df[['additional_requirements', 'education_speciality', 'other_vacancy_benefit', 'position_requirements', 'position_responsibilities', 'required_certificates']]



from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def remove_urls(text, replacement_text=""):
    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Use the sub() method to replace URLs with the specified replacement text
    text_without_urls = url_pattern.sub(replacement_text, text)

    return text_without_urls

import re


from pymorphy3 import MorphAnalyzer
import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords


stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def lemmatize(doc):
    patterns = "[0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-–•]+"
    doc = re.sub(patterns, ' ', doc)
    tokens = ''
    for token in doc.split():
        token = token.strip()
        token = morph.normal_forms(token)[0]
        if token not in stopwords_ru:
            tokens += (f' {token}')
    return tokens

def clear_data(input_str):
    if pd.isna(input_str):
        return input_str
    return lemmatize(remove_urls(strip_tags(input_str)))


counter = 0
for index, row in df.iterrows():

    names_for_clear = ['additional_requirements', 'education_speciality', 'other_vacancy_benefit', 'position_requirements', 'position_responsibilities', 'required_certificates']
    for i in names_for_clear:
        df.loc[index, i] = clear_data(row[i])
    if index % 10000 == 0:
        print(f'{index}/631117')
    if index % 50000 == 0:
        # Сохраняем в CSV файл
        df.to_csv(f'clear_data{counter}.csv', index=False, encoding='utf-8')
        print(f"сохранено!{counter}")
        counter += 1

df.to_csv(f'clear_data_fin.csv', index=False, encoding='utf-8')
print(f"завершено")