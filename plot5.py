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



need_dtypes = read_types("types\[5]asteroid.json")

df = pd.read_csv("datasets_for_graph\[5]asteroid.csv", dtype=need_dtypes)

print(df.dtypes)

plt.plot(df['diameter'])
plt.title('Линейный график диаметра астероидов')
plt.xlabel('Индекс астероида')
plt.ylabel('Диаметр')
plt.savefig('graph5/diametrs.png')
plt.close()



plt.figure(figsize=(8, 8))
categories = df['class'].value_counts()
merged_categories = categories[:5]
merged_categories['Other'] = categories[5:].sum()
merged_categories.plot(kind='pie', autopct='%1.1f%%')
plt.title('Распределение классов астероидов')
plt.ylabel('')
plt.savefig('graph5/asteroid_class.png')
plt.close()



plt.figure(figsize=(10, 6))
df.groupby('class')['diameter'].mean().plot(kind='line')
plt.title('Средний размер астероидов по классам')
plt.xlabel('Класс астероида')
plt.ylabel('Средний диаметр')
plt.savefig('graph5/mean_diameter_by_class.png')
plt.close()



sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="YlGnBu", cbar=False)
plt.savefig(r'graph5\corr.png')
plt.close()

sns.heatmap(df.select_dtypes(include=[np.number]).corr())
plt.savefig(r'graph5\corr2.png')
plt.close()

