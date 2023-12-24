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



need_dtypes = read_types("types\[3]flights.json")

df = pd.read_csv("datasets_for_graph\[3]flights.csv", dtype=need_dtypes)

print(df.dtypes)

#1
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title('Корреляционная матрица')
plt.savefig(r'graph3\corr1.png')
plt.close()

sns.heatmap(df.select_dtypes(include=[np.number]).corr())
plt.savefig(r'graph3\corr2.png')
plt.close()

sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cbar=False)
plt.savefig(r'graph3\corr3.png')
plt.close()


plt.pie(df['AIRLINE'].value_counts(), labels=df['AIRLINE'].unique())
plt.title('Круговая диаграмма авиакомпаний')
plt.savefig('круговая_диаграмма.png')
plt.savefig(r'graph3\airlines.png')
plt.close()


plt.hist(df['DISTANCE'], bins=20)
plt.title('Гистограмма расстояния')
plt.xlabel('Расстояние')
plt.ylabel('Частота')
plt.savefig('гистограмма.png')
plt.savefig(r'graph3\diatance.png')
plt.close()


plt.pie(df['DAY_OF_WEEK'].value_counts(), labels=["Sun", "Mon", "Tue", "Wen", "Thu", "Fri", "Sat"])
plt.title('Распределение полётов по дням')
plt.savefig(r'graph3\day_of_week.png')
plt.close()


mean_time = df.groupby('DAY_OF_WEEK')['ELAPSED_TIME'].mean()

plt.plot(["Sun", "Mon", "Tue", "Wen", "Thu", "Fri", "Sat"], mean_time.values)
plt.title('Среднее время в пути по дням недели')
plt.xlabel('День недели')
plt.ylabel('Среднее время в пути')
plt.savefig(r'graph3\avg_time_by_day.png')
plt.close()