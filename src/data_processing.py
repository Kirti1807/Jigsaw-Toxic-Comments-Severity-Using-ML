import numpy as np
import pandas as pd
import logging
from data_ingestion import DataUtils
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from application_logger import CustomApplicationLogger


class DataProcess:
    def __init__(self, data) -> None:
        self.data = data
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\DataProcessingsLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    def check_null_values(self):
        """ 
        Check for null values in dataframe 
        """
        self.logging.log(self.file_object, "Checking for null values in dataframe")
        try:
            print("Checking for null values in dataframe")
            print(self.data.isnull().sum())
        except Exception as e:
            self.logging.log(self.file_object, f"Error in checking null values: {e}")
            raise e

    def apply_all_processing_on_train_test_data(self):
        # apply all the Review processing methods on train and test data
        self.logging.log(
           self.file_object,  "Applying all the Review processing methods on train and test data"
        )
        try:
            self.data["text"] = self.data["text"].apply(
                lambda x: self.Review_processing(x)
            )
            self.data["text"] = self.data["text"].apply(
                lambda x: self.remove_punctuation(x)
            )
            self.data["text"] = self.data["text"].apply(
                lambda x: self.remove_numbers(x)
            )
            self.data["text"] = self.data["text"].apply(
                lambda x: self.remove_special_characters(x)
            )
            self.data["text"] = self.data["text"].apply(
                lambda x: self.remove_short_words(x)
            )
            self.data["text"] = self.data["text"].apply(
                lambda x: self.remove_stopwords(x)
            )
            self.data["text"] = self.data["text"].apply(lambda x: self.lemmatization(x))

            return self.data

        except Exception as e:
            self.logging.log(
               self.file_object,  "Error in applying all the Review processing methods on train and test data"
            )
            self.logging.log(self.file_object, e)
            return None

    def Review_processing(self, Review):
        self.logging.log(self.file_object, "Applying Review processing methods on train and test data")
        try:
            Review = Review.lower()
            Review = Review.replace("\n", " ")
            Review = Review.replace("\r", " ")
            Review = Review.replace("\t", " ")
            Review = Review.replace("\xa0", " ")
            Review = Review.replace("\u200b", " ")
            Review = Review.replace("\u200c", " ")
            Review = Review.replace("\u200d", " ")
            Review = Review.replace("\ufeff", " ")
            Review = Review.replace("\ufeef", " ")
        except Exception as e:
            self.logging.log(
                self.file_object, "Error in applying Review processing methods on train and test data"
            )
            self.logging.log(self.file_object, e)
            return None
        return Review

    def stemming(self, Review):
        self.logging.log(self.file_object, "Applying stemming methods on train and test data")
        try:
            Review = Review.split()
            ps = PorterStemmer()
            Review = [ps.stem(word) for word in Review]
            Review = " ".join(Review)
        except Exception as e:
            self.logging.log(
                self.file_object, "Error in applying stemming methods on train and test data"
            )
            self.logging.log(self.file_object, e)
            return None
        return Review

    def lemmatization(self, Review):
        Review = Review.split()
        lem = WordNetLemmatizer()
        Review = [lem.lemmatize(word) for word in Review]
        Review = " ".join(Review)
        return Review

    def remove_stopwords(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = " ".join(Review)
        return Review

    def remove_punctuation(self, Review):
        # remove all punctuation except full stop, exclaimation mark and question mark
        Review = Review.split()
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)

        return Review

    def remove_numbers(self, Review):
        Review = Review.split()
        Review = [word for word in Review if not word.isnumeric()]
        Review = " ".join(Review)
        return Review

    def remove_special_characters(self, Review):
        Review = Review.split()
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)
        return Review

    def remove_short_words(self, Review):
        Review = Review.split()
        Review = [word for word in Review if len(word) > 2]
        Review = " ".join(Review)
        return Review

    def remove_stopwords_and_punctuation(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = [word for word in Review if word.isalpha()]
        Review = " ".join(Review)
        return Review

    def remove_stopwords_and_punctuation_and_numbers(self, Review):
        Review = Review.split()
        stop_words = set(stopwords.words("english"))
        Review = [word for word in Review if not word in stop_words]
        Review = [word for word in Review if word.isalpha()]
        Review = [word for word in Review if not word.isnumeric()]
        Review = " ".join(Review)
        return Review

    def remove_nan_values(self, df):
        # fill nan values with UNKOWN and return the dataframe
        df = df.fillna("UNKOWN")
        return df


class DataValidation:
    def __init__(self, data) -> None:
        self.data = data

    def data_splitting(self):
        self.logging.log(self.file_object, "Data Splitting")
        try:
            X = self.data.drop("y", axis=1)
            Y = self.data["y"]

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            self.logging.log(self.file_object, f"Error loading data: {e}")
            raise e

    # def split_data(self):
    #     self.logging.log("Splitting the data")
    #     try:
    #         list_of_df = np.array_split()


if __name__ == "__main__":
    # data_utils = DataUtils()
    # jigsaw_data, c3_data, ruddit_data = data_utils.load_different_data()
    # Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
    # data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
    # data_process = DataProcess(data, jigsaw_data, c3_data, ruddit_data)
    # data_process.check_null_values()
    # data_processed = data_process.apply_all_processing_on_train_test_data()
    # data_processed.head()
    pass

