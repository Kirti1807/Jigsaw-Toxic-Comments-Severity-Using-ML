import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from application_logger import CustomApplicationLogger


class DataUtils:
    def __init__(self) -> None:
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\DataIngestionLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    def load_different_data(self):
        self.logging.log(
            self.file_object,
            "In load_different_data method in DataUtils Class  : Loading Jigsaw, C3 and ruddit data",
        )
        try:

            Jigsaw = pd.read_csv(
                r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\AllDataRaw\train_data.csv"
            )
            c3_data = pd.read_csv(
                r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\AllDataRaw\C3_anonymized.csv"
            )
            ruddit_data = pd.read_csv(
                r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\AllDataRaw\ruddit_with_text.csv"
            )
            self.logging.log(self.file_object, "Data loaded successfully")
            return Jigsaw, c3_data, ruddit_data

        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In load_different_data method in DataUtils Class  : Error loading data: {e}",
            )
            raise e

    def prepare_data(self):
        self.logging.log(
            self.file_object,
            "In prepare_data method in DataUtils Class :Preparing data for Further processing",
        )
        try:
            Jigsaw, c3_data, ruddit_data = self.load_different_data()
            Jigsaw.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1, inplace=True)
            # drop all the columns except comment_text and agree_toxicity_expt
            c3_data = c3_data[["comment_text", "agree_toxicity_expt"]]
            ruddit_data = ruddit_data[["txt", "offensiveness_score"]]
            # rename the columns
            c3_data.columns = ["text", "y"]
            ruddit_data.columns = ["text", "y"]
            # scaling all target values between 0-1
            scaler = MinMaxScaler()
            ruddit_data["y"] = scaler.fit_transform(ruddit_data[["y"]])
            c3_data["y"] = scaler.fit_transform(c3_data[["y"]])
            Jigsaw["y"] = scaler.fit_transform(Jigsaw[["y"]])
            self.logging.log(
                self.file_object,
                "In prepare_data method in DataUtils Class : Data prepared successfully",
            )
            return Jigsaw, c3_data, ruddit_data
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In prepare_data method in DataUtils Class  : Error preparing data: {e}",
            )
            raise e

    def concatenate_dfs(self, Jigsaw, c3_data, ruddit_data):
        self.logging.log(
            self.file_object,
            "In concatenate_dfs method in DataUtils Class : Concatenating dataframes",
        )
        try:
            # concatenate all the dataframes
            data = pd.concat([Jigsaw, c3_data, ruddit_data], ignore_index=True)
            self.logging.log(
                self.file_object,
                "In concatenate_dfs method in DataUtils Class : concatenated successfully",
            )
            return data
        except Exception as e:
            self.logging.log(
                self.file_object,
                f"In concatenate_dfs method in DataUtils Class : Error concatenating data: {e}",
            )
            raise e

        

if __name__ == "__main__":
    data_utils = DataUtils()
    jigsaw_data, c3_data, ruddit_data = data_utils.load_different_data()
    Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
    data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
    # print('jigsaw:',Jigsaw.shape)
    # print(Jigsaw.head())
    # print('c3:', c3_data.shape)
    # print(c3_data.head())
    # print('ruddit:',ruddit_data.shape)
    # print(ruddit_data.head(20))

