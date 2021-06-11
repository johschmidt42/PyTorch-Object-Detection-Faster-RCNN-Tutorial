from utils import get_filenames_of_path
import pathlib
import torch
import pandas as pd

# inputs
root_path = r'C:\Users\johan\Desktop\JuanPrada\Clone6 +_- dTAG - Images\results'
folder_names = ['Rep1 + WT', 'Rep2', 'Rep3', 'Rescue Experiment March2021']

# batch processing
dataframes = []

for folder_name in folder_names:
    inputs = get_filenames_of_path(pathlib.Path(root_path) / folder_name)
    inputs.sort()

    dataframes_per_folder = []

    for prediction_path in inputs:
        pred = torch.load(prediction_path)
        scores_labels = {key: value for key, value in pred.items() if key in ['scores', 'labels']}
        df = pd.DataFrame(scores_labels)

        data_series = df.labels.value_counts()
        data_series = data_series.append(pd.Series({'total': df.labels.size}))
        data_series.index = ['open', 'closed', 'total']
        data_series = data_series.to_frame(prediction_path.stem)
        data_series = data_series.T
        dataframes_per_folder.append(data_series)
        print(data_series, '\n')

    dataframes.append(pd.concat(dataframes_per_folder))

from pandas import ExcelWriter

excel_path = pathlib.Path(root_path).parent / 'results.xlsx'


def save_xls(dataframes, excel_path):
    with ExcelWriter(excel_path) as writer:
        for folder_name, dataframe in zip(folder_names, dataframes):
            dataframe.to_excel(writer, folder_name)
        writer.save()


save_xls(dataframes, excel_path)
