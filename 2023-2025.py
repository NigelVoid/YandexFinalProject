import pandas as pd
import os


def process(df):
    # убираем строки в которых нет критически важных данных
    df.dropna(subset=["started_at", "ended_at"], inplace=True)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])

    df["ride_length_seconds"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()

    # если станция не указана
    df['start_station_name'].fillna('Вне станции')
    df['end_station_name'].fillna('Вне станции')

    # убираем невозможные длительности поездки и ложные старты
    df = df[(df["ride_length_seconds"] >= 180) & (df["ride_length_seconds"] <= 86400)]
    return df


years_dfs = []

for year in ["2023", "2024", "2025"]:
    list_dfs = []
    for csv_file in os.listdir(year):
        list_dfs.append(process(pd.read_csv(year + "/" + csv_file)))

    year_df = pd.concat(list_dfs, ignore_index=True)
    year_df["year"] = year
    years_dfs.append(year_df)

final_df = pd.concat(years_dfs, ignore_index=True)
final_df.to_csv("2023-2025.csv", index=False)



