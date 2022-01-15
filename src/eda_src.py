import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from wordcloud import WordCloud, STOPWORDS
from application_logger import CustomApplicationLogger

from data_ingestion import DataUtils


class EDA:
    def __init__(self, data, jigsaw, c3data, ruddit) -> None:
        """ 
        Input: 
            data: dataframe ( concatenated with Jigsaw, C3 and ruddit data) 
            jigsaw: dataframe (Jigsaw data) 
            c3data: dataframe (C3 data) 
            ruddit: dataframe (ruddit data)
        """
        self.data = data
        self.jigsaw = jigsaw
        self.c3data = c3data
        self.ruddit = ruddit
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\EDALogs.txt", "a+",
        )
        self.logging = CustomApplicationLogger()

    def data_info(self):
        """ 
        Basic Data Exploration like shape, info and etc 
        """
        self.logging.log(self.file_object, "Basic Data Exploration")
        try:
            print("Shape of the data: ", self.data.shape)
            print("Info of the data: ", self.data.info())
            print("Data types: ", self.data.dtypes)
            print("Data head: ", self.data.head())
        except Exception as e:
            self.logging.log(self.file_object, f"Error in basic data exploration: {e}")
            raise e

    def exploring_distributions(self):
        """ 
        distributions of target labels 
        """
        # exploring the distribution of target labels
        self.logging.log(self.file_object, "Exploring the distribution of target labels")
        try:
            print("Showing Distribution of concatenated data")
            sns.distplot(self.data["y"])
            plt.show()
            print("Showing Distribution of Jigsaw data")
            sns.distplot(self.jigsaw["y"])
            plt.show()
            print("Showing Distribution of C3 data")
            sns.displot(self.c3data["y"])
            plt.show()
            print("Showing Distribution of ruddit data")
            sns.distplot(self.ruddit["y"])
            plt.show()
        except Exception as e:
            self.logging.log(self.file_object, f"Error in exploring distribution: {e}")
            raise e

    def word_cloud(self):
        """ 
        WordCloud 
        """
        self.logging.log(self.file_object, "WordCloud of the data")
        try:
            print("WordCloud of the data")
            # Generate a word cloud image
            wordcloud = WordCloud(
                background_color="white",
                stopwords=STOPWORDS,
                max_words=200,
                max_font_size=40,
                random_state=42,
            ).generate(str(self.data["text"]))

            # Display the generated image:
            plt.figure(figsize=(20, 20))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            print("WordCloud of the Jigsaw data")
            wordcloud = WordCloud(
                background_color="white",
                stopwords=STOPWORDS,
                max_words=200,
                max_font_size=40,
                random_state=42,
            ).generate(str(self.jigsaw["text"]))

            # Display the generated image:
            plt.figure(figsize=(20, 20))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            print("WordCloud of the C3 data")
            wordcloud = WordCloud(
                background_color="white",
                stopwords=STOPWORDS,
                max_words=200,
                max_font_size=40,
                random_state=42,
            ).generate(str(self.c3data["text"]))

            # Display the generated image:
            plt.figure(figsize=(20, 20))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show()
            print("WordCloud of the ruddit data")
            wordcloud = WordCloud(
                background_color="white",
                stopwords=STOPWORDS,
                max_words=200,
                max_font_size=40,
                random_state=42,
            ).generate(str(self.ruddit["text"]))

            # Display the generated image:
            plt.figure(figsize=(20, 20))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.show() 
            
        except Exception as e:
            self.logging.log(self.file_object, f"Error in word cloud: {e}")
            raise e

    def skeweness_and_kurtosis(self):
        self.logging.log(self.file_object, "Skewness and Kurtosis of the data")
        try:
            print("Skewness of the data")
            print(self.data.skew())
            print("Kurtosis of the data")
            print(self.data.kurt())
            print("Skewness of the Jigsaw data")
            print(self.jigsaw.skew())
            print("Kurtosis of the Jigsaw data")
            print(self.jigsaw.kurt())
            print("Skewness of the C3 data")
            print(self.c3data.skew())
            print("Kurtosis of the C3 data")
            print(self.c3data.kurt())
            print("Skewness of the ruddit data")
            print(self.ruddit.skew())
            print("Kurtosis of the ruddit data")
            print(self.ruddit.kurt())
        except Exception as e:
            self.logging.log(self.file_object, f"Error in skewness and kurtosis: {e}")
            raise e


if __name__ == "__main__":
    # =======================================================================
    # Load data
    # data_utils = DataUtils()
    # jigsaw_data, c3_data, ruddit_data = data_utils.load_different_data()
    # Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
    # data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
    # =======================================================================

    # =======================================================================
    # Exploring data
    # eda = EDA(data, jigsaw_data, c3_data, ruddit_data)
    # eda.data_info()
    # eda.exploring_distributions()
    # eda.word_cloud()
    # eda.skeweness_and_kurtosis()

    # ========================================================================
    pass

