import pandas as pd
import numpy as np
import json
import os


#Ссылка на 6 dataset https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022?select=Combined_Flights_2022.csv

def read_csv(filename: str, columns: list[str] = []):
    if(len(columns) == 0):
        return pd.read_csv(filename, chunksize=700_000)
    return pd.read_csv(filename,  chunksize=700_000, usecols=columns)


def write_data_into_json(file_name, data):
    with open(file_name + ".json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4, default=str))

def write_memory_stat_by_file(filename, path: str = r"start_memory_stat\memory_stat_"):
    memory_stat = {
        "total_memory_usage": 0
    }
    column_stat = dict()
    memory_stat["file_size"] = str(os.path.getsize(filename) // 1024)
    chank = pd.read_csv(filename, chunksize=1).get_chunk()
    for title in chank.columns:
        column_stat[title] = {
                    "memory_abs": 0,
                    "type": chank.dtypes[title]
                }
    for chunck in pd.read_csv(filename, chunksize=900_000):
        memory_usage_columns = chunck.memory_usage(deep=True)
        memory_stat["total_memory_usage"] += memory_usage_columns.sum()
        keys = chunck.dtypes.keys()
        for key in chunck.dtypes.keys():
            column_stat[key]["memory_abs"] += memory_usage_columns[key]
    memory_stat["total_memory_usage"] //= 1024
    for key in column_stat.keys():
        column_stat[key]["memory_abs"] //= 1024
        column_stat[key]["memory_per"] = round(column_stat[key]["memory_abs"] / memory_stat["total_memory_usage"] * 100, 4)
    column_stat = dict(sorted(column_stat.items(), key=lambda item: item[1]["memory_abs"],reverse=True))
    memory_stat.update(column_stat)
    write_data_into_json(path + filename.replace("\\",".").replace("/",".").split(".")[1], memory_stat)


def write_memory_stat_by_df(filename, df: pd.DataFrame, path: str = r"start_memory_stat\memory_stat_"):
    memory_stat = {
        "total_memory_usage": 0
    }
    column_stat = dict()
    for title in df.columns:
        column_stat[title] = {
                    "memory_abs": 0,
                    "type": df.dtypes[title]
                }
    memory_usage_columns = df.memory_usage(deep=True)
    memory_stat["total_memory_usage"] += memory_usage_columns.sum()
    keys = df.dtypes.keys()
    for key in keys:
        column_stat[key]["memory_abs"] += memory_usage_columns[key]

    memory_stat["total_memory_usage"] //= 1024
    for key in column_stat.keys():
        column_stat[key]["memory_abs"] //= 1024
        column_stat[key]["memory_per"] = round(column_stat[key]["memory_abs"] / memory_stat["total_memory_usage"] * 100, 4)
    column_stat = dict(sorted(column_stat.items(), key=lambda item: item[1]["memory_abs"],reverse=True))
    memory_stat.update(column_stat)
    write_data_into_json(path + filename.replace("\\",".").replace("/",".").split(".")[1], memory_stat)



def mem_usage(pandas_obj):
    if(isinstance(pandas_obj, pd.DataFrame)):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)

def optimization(filename: str):
    size = os.stat(filename).st_size
    dataset = pd.DataFrame()
    optimize_dataset = pd.DataFrame()
    if(size < 300000000):
        dataset = read_csv(filename).get_chunk()
        optimize_dataset = dataset.copy()
    else:
        columns = get_selected_columns(filename)
        for chank in read_csv(filename, columns=columns):
            dataset = pd.concat([dataset, chank])
    

    optimize_dataset = dataset.copy()
    converted_obj = opt_obj(dataset)
    converted_int = opt_int(dataset)
    converted_float = opt_float(dataset)

    optimize_dataset[converted_obj.columns] = converted_obj
    optimize_dataset[converted_int.columns] = converted_int
    optimize_dataset[converted_float.columns] = converted_float


    write_memory_stat_by_df("reloaded_dataset/" + filename.replace("\\",".").replace("/",".").split(".")[1] + ".csv",
                            dataset, path= r"reload_memory_stat/")
    write_memory_stat_by_df("opt_dataset/" + filename.replace("\\",".").replace("/",".").split(".")[1] + ".csv",
                            optimize_dataset, path= r"opt_memory_stat/")
    





    # print(dataset.dtypes[columns])
    #print(optimize_dataset.dtypes[columns])
    print(optimize_dataset.dtypes)
    print(mem_usage(dataset))
    print(mem_usage(optimize_dataset))


    if(size < 300000000):
        write_data_into_json("types/" + filename.replace("\\",".").replace("/",".").split(".")[1], optimize_dataset.dtypes[get_selected_columns(filename)].to_dict())
        optimize_dataset[get_selected_columns(filename)].to_csv("datasets_for_graph/" + filename.replace("\\",".").replace("/",".").split(".")[1] + ".csv", index = False)
    else:
        optimize_dataset.to_csv("datasets_for_graph/" + filename.replace("\\",".").replace("/",".").split(".")[1] + ".csv", index = False)
        write_data_into_json("types/" + filename.replace("\\",".").replace("/",".").split(".")[1], optimize_dataset.dtypes.to_dict())

    #return optimize_dataset



def opt_obj(df: pd.DataFrame):
    result = pd.DataFrame()
    dataset = df.select_dtypes(include=["object"]).copy()
    for column_name in dataset:
        column = dataset[column_name]
        num_unique = len(column.unique())
        column_len = len(column)
        if(num_unique / column_len < 0.5):
            result.loc[ : , column_name] = column.astype("category")
        else:
            result.loc[ : , column_name] = column

    # print(mem_usage(dataset))
    # print(mem_usage(result))
    return result





def opt_int(df: pd.DataFrame):
    result = pd.DataFrame()
    dataset_int = df.select_dtypes(include=["int"])
    #list(map(lambda x: print(dataset_int[x].unique()), dataset_int))
    #print(dataset_int)
    converted_int = dataset_int.apply(pd.to_numeric, downcast="unsigned")
    # print(mem_usage(dataset_int))
    # print(mem_usage(converted_int))

    compare_ints = pd.concat([dataset_int.dtypes, converted_int.dtypes], axis=1)
    compare_ints.columns = ["before", "after"]
    compare_ints.apply(pd.Series.value_counts)
    # print(compare_ints)

    return converted_int 

def opt_float(df: pd.DataFrame):
    dataset_float = df.select_dtypes(include=["float"])

    converted_float = dataset_float.apply(pd.to_numeric, downcast="float")
    # print(mem_usage(dataset_float))
    # print(mem_usage(converted_float))

    compare_floats = pd.concat([dataset_float.dtypes, converted_float.dtypes], axis=1)
    compare_floats.columns = ["before", "after"]
    compare_floats.apply(pd.Series.value_counts)
    # print(compare_floats)

    return converted_float

def save_columns(df: pd.DataFrame, column_names: list[str], filename: str):
    need_column = dict()
    opt_dtypes = df.dtypes
    for key in column_names:
        need_column[key] = opt_dtypes[key]
        # print(f"{key}: {opt_dtypes[key]}")
    has_header = True
    for chunck in pd.read_csv(filename,
                              usecols=lambda x: x in column_names,
                              dtype=need_column,chunksize=500_000):
        # print(mem_usage(chunck))
        chunck.to_csv(r"datasets_for_graph\\" + filename.replace("\\", ".").split(".")[1] + "_selected.csv", mode="a", header= has_header, index = False)
        has_header = False


def get_file_names(path):
    return list(map(lambda file: "dataset_6\\" + file,os.listdir(path=path)))

def get_selected_columns(filename: str):
    match filename:
        case "dataset_6\[1]game_logs.csv":
            return ["date","day_of_week","h_game_number","v_walks",
                    "length_outs","v_score","v_individual_earned_runs",
                    "v_name","length_minutes","h_errors"]
        case "dataset_6\[2]automotive.csv.zip":
            return ["lastSeen","msrp","isNew","color",
                    "brandName","modelName","firstSeen",
                    "stockNum","vin","askPrice"]
        case "dataset_6\[3]flights.csv":
            return ["YEAR","MONTH","DAY","DAY_OF_WEEK",
                    "AIRLINE","SCHEDULED_ARRIVAL","SCHEDULED_TIME",
                    "ELAPSED_TIME","DISTANCE","TAIL_NUMBER"]
        case "dataset_6\[4]vacancies.csv.gz":
            return ["id","employment_name","schedule_name","experience_name",
                    "name","area_name","published_at",
                    "prof_classes_found","salary_from","salary_currency"]
        case "dataset_6\[5]asteroid.zip":
            return ["id","spkid","full_name","name",
                    "diameter","per","sigma_tp",
                    "class","diameter","albedo"]
        case _:
            return ["FlightDate","ArrTime","Distance","OriginStateName",
                    "DestStateName","CRSArrTime","Airline",
                    "Cancelled","Diverted","Dest"]

def check_and_create_dir():
    for dir in ["datasets_for_graph"]:
        if not os.path.exists(dir):
            os.makedirs(dir)

if __name__ == "__main__":
    files = get_file_names("dataset_6") #файлы считываются из папка dataset_6
    check_and_create_dir()
    #После чего проходимся по всем файлам, можно проходить по одному файлу, если указать срез, например file[0 : 1] (Combined_Flights_2022.csv0 и т.д.
    for file in files:
        #dataset = read_csv(file)
        write_memory_stat_by_file(file)
        optimization(file)


