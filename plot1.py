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



need_dtypes = read_types("types\[1]game_logs.json")

df = pd.read_csv("datasets_for_graph\[1]game_logs.csv", dtype=need_dtypes)

print(df.dtypes)


counts = df['day_of_week'].value_counts()
plt.pie(counts, labels=counts.index)
plt.title('Распределение игр по дням недели')

plt.savefig(r'graph1\day_of_weeks.png')
plt.close()





grouped_data = df.groupby('day_of_week')['v_score'].mean()


plt.bar(grouped_data.index, grouped_data)


plt.xlabel('День недели')  
plt.ylabel('Средний v_score') 
plt.title('Зависимость среднего v_score от дня недели')  

plt.savefig(r'graph1\day_of_week_by_v_score.png')
plt.close()



plt.scatter(df['length_minutes'], df['h_errors'])

plt.xlabel('Длительность (минуты)')
plt.ylabel('Количество ошибок')
plt.title('Зависимость ошибок от длительности игры')

plt.savefig(r'graph1\day_of_week_by_v_score.png')
plt.close()



day_of_week_counts = df['day_of_week'].value_counts()

plt.bar(day_of_week_counts.index, day_of_week_counts.values)
plt.xlabel('Day of Week')
plt.ylabel('Number of Games')
plt.title('Games by Day of Week')
plt.savefig(r'graph1\games_by_day_of_week_by_v_score.png')
plt.close()





plt.scatter(df['v_walks'], df['h_errors'], color='red')
plt.xlabel('Visiting Team Walks')
plt.ylabel('Home Team Errors')
plt.title('Relationship between Walks and Errors')
plt.savefig(r'graph1\v_walks_by_h_errors.png')
plt.close()



plt.hist(df['length_minutes'], bins=100, edgecolor='black')
plt.xlabel('Game Length (Minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Game Lengths')
plt.savefig(r'graph1\length_minutes.png')
plt.close()



sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="YlGnBu", cbar=False)
plt.savefig(r'graph1\corr.png')
plt.close()


