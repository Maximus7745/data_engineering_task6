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



need_dtypes = read_types("types\[2]automotive.json")

df = pd.read_csv("datasets_for_graph\[2]automotive.csv", dtype=need_dtypes)

print(df.dtypes)

#1
plt.figure(figsize=(10, 6))
df['lastSeen'] = pd.to_datetime(df['lastSeen'])
df['weekday'] = df['lastSeen'].dt.weekday
df['weekday'].value_counts().sort_index().plot(kind='line')
plt.xticks(range(7), ["Sun", "Mon", "Tue", "Wen", "Thu", "Fri", "Sat"])
plt.xlabel('День недели')
plt.ylabel('Количество авто')
plt.title('Распределение автомобилей по дням недели просмотра')
plt.savefig(r'graph2\car_by_day.png')
plt.close()


#2
df['isNew'].value_counts().plot(kind='pie')
plt.ylabel('')
plt.title('Новые и подержаные машины')
plt.savefig(r'graph2\new_car.png')
plt.close()

#3
plt.figure(figsize=(16, 14))
df['brandName'].value_counts().head(20).plot(kind='bar')
plt.xlabel('Марка автомобиля')
plt.ylabel('Количество')
plt.title('Распределение автомобилей по маркам')
plt.savefig('graph2/brand_distribution.png')
plt.close()

#4
top_10_colors = df['color'].value_counts().nlargest(10)
other_colors_count = df['color'].value_counts().sum() - top_10_colors.sum()
plt.figure(figsize=(8, 8))
color_counts = list(top_10_colors.values) + [other_colors_count]
colors = list(top_10_colors.index) + ['Other']
plt.pie(color_counts, labels=colors, autopct='%1.1f%%')
plt.title('Распределение цветов автомобилей')
plt.savefig('graph2/color_distribution.png')
plt.close()

#5
plt.figure(figsize=(10, 6))
plt.scatter(df['msrp'], df['askPrice'], alpha=0.5)
plt.xlabel('Ориентировочная Цена')
plt.ylabel('Цена Запроса')
plt.title('Корреляция между Ориентировочной Ценой и Ценой Запроса')
plt.savefig('graph2/price_correlation.png')
plt.close()

#6

plt.figure(figsize=(10, 6))
plt.boxplot(df['msrp'])
plt.xlabel('Цена')
plt.ylabel('Размах')
plt.title('Разброс цен на автомобили')
plt.savefig('graph2/price_range.png')
plt.close()