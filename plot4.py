import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def read_types(filename: str):
    dtypes = {}
    with open(filename, "r") as f:
        dtypes = json.load(f)

    for key in dtypes.keys():
        if(dtypes[key] == "category"):
            dtypes[key] = pd.CategoricalDtype
        else:
            dtypes[key] = np.dtype(dtypes[key])

    return dtypes



need_dtypes = read_types("types\[4]vacancies.json")

df = pd.read_csv("datasets_for_graph\[4]vacancies.csv", dtype=need_dtypes)

print(df.dtypes)

#1
plt.figure(figsize=(10, 6))
df['schedule_name'].value_counts().plot(kind='bar')
plt.xlabel('График работы')
plt.ylabel('Количество вакансий')
plt.title('Распределение количества вакансий по графику работы')
plt.savefig('graph4/schedule_vacancies.png')
plt.close()


#2
plt.figure(figsize=(10, 6))
df.groupby('experience_name')['salary_from'].mean().plot(kind='bar')
plt.xlabel('Опыт работы')
plt.ylabel('Средний размер зарплаты')
plt.title('Распределение зарплатной вилки по опыту работы')
plt.savefig('graph4/salary_experience.png')
plt.close()

#3
plt.figure(figsize=(8, 8))
df['employment_name'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Распределение типов занятости')
plt.ylabel('')
plt.savefig('graph4/employment_distribution.png')
plt.close()

#4
plt.figure(figsize=(10, 10))
df.groupby('area_name')['salary_from'].mean().sort_values(key= lambda x: -x).head(35).plot(kind='bar')
plt.xlabel('Регион')
plt.ylabel('Средний размер зарплаты')
plt.title('Распределение зарплаты по регионам')
plt.savefig('graph4/salary_by_region.png')
plt.close()

#5
plt.figure(figsize=(10, 6))
df['area_name'].value_counts().head(35).plot(kind='bar')
plt.xlabel('Регион')
plt.ylabel('Количество вакансий')
plt.title('Распределение числа вакансий по регионам')
plt.savefig('graph4/vacancies_by_region.png')
plt.close()
#6

plt.figure(figsize=(10, 6))
df['prof_classes_found'].value_counts().head(10).plot(kind='bar')
plt.xlabel('Категория профессии')
plt.ylabel('Количество вакансий')
plt.title('Распределение вакансий по категориям профессий')
plt.savefig('graph4/profession_categories.png')
plt.close()

#7
plt.figure(figsize=(10, 6))
df['experience_name'].value_counts().plot(kind='bar')
plt.xlabel('Уровень опыта')
plt.ylabel('Количество вакансий')
plt.title('Распределение вакансий по уровню опыта')
plt.savefig('graph4/experience_level.png')
plt.close()