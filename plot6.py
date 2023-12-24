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



need_dtypes = read_types("types\Combined_Flights_2022.json")

df = pd.read_csv("datasets_for_graph\Combined_Flights_2022.csv", dtype=need_dtypes)

print(df.dtypes)






plt.figure(figsize=(10, 6))
df['FlightDate'] = pd.to_datetime(df['FlightDate'])
df['FlightDate'].value_counts().sort_index().plot(kind='line')
plt.xlabel('Дата')
plt.ylabel('Количество полетов')
plt.title('Изменение числа полетов по датам')
plt.savefig('graph6/flight_count_line.png')
plt.close()



plt.figure(figsize=(10, 6))
df['OriginStateName'].value_counts().plot(kind='bar')
plt.xlabel('Штат отправления')
plt.ylabel('Количество полетов')
plt.title('Распределение полетов по штатам отправления')
plt.savefig('graph6/flights_by_origin_state.png')
plt.close()



plt.figure(figsize=(8, 8))
df['Airline'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Распределение полетов по авиалиниям')
plt.ylabel('')
plt.savefig('graph6/flights_by_airline.png')
plt.close()



plt.figure(figsize=(10, 6))
df['Distance'].plot(kind='hist', bins=20)
plt.xlabel('Расстояние полета (мили)')
plt.ylabel('Количество полетов')
plt.title('Распределение расстояния полетов')
plt.savefig('graph6/distance_distribution.png')
plt.close()



plt.figure(figsize=(10, 6))
df.loc[df['Cancelled'] == False]['DestStateName'].value_counts().plot(kind='bar')
plt.xlabel('Штат назначения')
plt.ylabel('Количество задержек')
plt.title('Распределение задержек по штатам назначения')
plt.savefig('graph6/delays_by_destination_state.png')
plt.close()



plt.figure(figsize=(8, 8))
df['Cancelled'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Распределение отмененных и перенаправленных рейсов')
plt.ylabel('')
plt.savefig('graph6/cancelled_diverted_pie.png')
plt.close()




numeric_cols = ['ArrTime', 'Distance']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляция числовых переменных')
plt.savefig('graph6/correlation_heatmap.png')
plt.close()




plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Cancelled'], y=df['ArrTime'])
plt.xlabel('Отмененные и неперенаправленные рейсы')
plt.ylabel('Время прибытия')
plt.title('Распределение времени прибытия по категории рейсов')
plt.savefig('graph6/cancelled_arrival_time_boxplot.png')
plt.close()

