import numpy as np
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk import sent_tokenize, word_tokenize
from application_logger import CustomApplicationLogger
from data_ingestion import DataUtils
from data_processing import DataProcess
import gensim
import re


class FeatureEngineering:
    def __init__(self, data) -> None:
        self.data = data
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\FeatureEngineeringLogs.txt",
            "a+",
        )
        self.logging = CustomApplicationLogger()

    def get_count_of_words(self):
        # get count of words in data
        try:
            self.logging.log(self.file_object, "Get count of words in data")
            word_count = self.data["text"].apply(lambda x: len(x.split()))
            self.logging.log(self.file_object, "Completed count of words")
            return word_count
        except Exception as e:
            self.logging.log(self.file_object, f"Error in basic data exploration: {e}")
            raise e

    def get_count_of_sentences(self):
        # get count of sentences in data
        try:
            self.logging.log(self.file_object, "Get count of sentences in data")
            sentences_count = self.data["text"].apply(lambda x: len(sent_tokenize(x)))
            self.logging.log(
                self.file_object, "Completed Get count of sentences in data"
            )
            return sentences_count
        except Exception as e:
            self.logging.log(self.file_object, f"Error in basic data exploration: {e}")
            raise e

    def get_average_word_length(self):
        try:
            # get average word length in data
            self.logging.log(self.file_object, "Get average word length in data")
            average_word_length = self.data["text"].apply(
                lambda x: np.mean([len(word) for word in x.split()])
            )
            self.logging.log(
                self.file_object, "Completed Get count of sentences in data"
            )
            return average_word_length
        except Exception as e:
            self.logging.log(self.file_object, f"Error in basic data exploration: {e}")
            raise e

    def get_average_sentence_length(self):
        # get average sentence length in data
        try:
            logging.info("Get average sentence length in data")
            average_sentence_length = self.data["text"].apply(
                lambda x: np.mean([len(sentence) for sentence in sent_tokenize(x)])
            )
            return average_sentence_length
        except Exception as e:
            self.logging.log(self.file_object, f"Error in basic data exploration: {e}")
            raise e

    def get_average_sentence_complexity(self):
        # get average sentence complexity in data
        logging.info("Get average sentence complexity in data")
        average_sentence_complexity = self.data["text"].apply(
            lambda x: np.mean(
                [len(word_tokenize(sentence)) for sentence in sent_tokenize(x)]
            )
        )
        return average_sentence_complexity

    def get_average_word_complexity(self):
        # get average word complexity in data
        logging.info("Get average word complexity in data")
        average_word_complexity = self.data["text"].apply(
            lambda x: np.mean([len(word_tokenize(word)) for word in x.split()])
        )
        return average_word_complexity

    def add_features(self):

        self.logging.log(self.file_object, "Add features")
        word_count = self.get_count_of_words()
        sentences_count = self.get_count_of_sentences()
        average_word_length = self.get_average_word_length()
        average_sentence_length = self.get_average_sentence_length()
        average_sentence_complexity = self.get_average_sentence_complexity()
        average_word_complexity = self.get_average_word_complexity()

        self.data["count_of_words"] = word_count
        self.data["count_of_setences"] = sentences_count
        self.data["average_word_length"] = average_word_length
        self.data["average_sentence_length"] = average_sentence_length
        self.data["average_sentence_complexity"] = average_sentence_complexity
        self.data["average_word_complexity"] = average_word_complexity
        self.logging.log(self.file_object, "Done Add features")
        return self.data

    def train_a_gensim_model(self):
        # train a gensim model
        logging.info("Train a gensim model")

        review_text = self.train_data.Review.apply(gensim.utils.simple_preprocess)
        model = gensim.models.Word2Vec(window=10, min_count=2, workers=4)
        model.build_vocab(review_text, progress_per=1000)
        model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(r"E:\Hackathon\UGAM\src\saved_model\ugam_reviews.model")
        return model

    def get_word_embeddings(self, model):
        # get word embeddings
        logging.info("Get word embeddings")
        word_embeddings = model.wv
        return word_embeddings

    def get_similar(self, word, model):
        if word in model.wv:
            return model.wv.most_similar(word)[0]  # try runnign again
        else:
            return None

    def make_acolumn(self, model):
        # make a new column "most similar words" and get the most similar words for every word in review text
        logging.info(
            "Make a new column 'most similar words' and get the most similar words for every word in review text and leave the word whic is not present in the model"
        )
        #
        self.train_data["most_similar_words"] = self.train_data["Review"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in review text
        )
        self.test_data["most_similar_words"] = self.test_data["Review"].apply(
            lambda x: [
                self.get_similar(word, model) for word in word_tokenize(x)
            ]  # get the most similar words for every word in review text
        )
        return self.train_data, self.test_data

    def process_most_similar_words(self, text):

        # process most similar words
        logging.info("Process most similar words")
        # process the column most similar words row by row
        # tokenize the word
        text = word_tokenize(text)
        for j in text:
            if j.isalpha() == False:
                text.remove(j)
            if j == "None":
                text.remove(j)
            if j == "":
                text.remove(j)
                # convert str to int
            if j.isdigit():
                text.remove(j)
        text = " ".join(text)
        # remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # remove numbers, None and empty strings
        text = re.sub(r"\d+", "", text)
        # remove None from text
        text = re.sub(r"None", "", text)
        # remove extra spaces
        text = re.sub(r"\s+", " ", text)
        # remove stop words
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        text = [word for word in text.split() if word not in stop_words]
        # convert list to str
        text = " ".join(text)
        return text


class Vectorization:
    def __init__(self, df) -> None:
        self.df = df

    def vectorize(self) -> pd.DataFrame:
        """ 
        Only vectorize concatenated data 
        """
        vectorizer = TfidfVectorizer(max_features=5000)

        extracted_data = list(vectorizer.fit_transform(self.df["text"]).toarray())
        extracted_data = pd.DataFrame(extracted_data)
        extracted_data.head()
        extracted_data.columns = vectorizer.get_feature_names()

        vocab = vectorizer.vocabulary_
        mapping = vectorizer.get_feature_names()
        keys = list(vocab.keys())

        extracted_data.shape
        Modified_df = extracted_data.copy()
        print(Modified_df.shape)
        Modified_df.head()
        Modified_df.reset_index(drop=True, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        Final_Training_data = pd.concat([self.df, Modified_df], axis=1)

        Final_Training_data.head()
        print(Final_Training_data.shape)
        Final_Training_data.drop(["text"], axis=1, inplace=True)
        Final_Training_data.head()
        Final_Training_data.to_csv("Final_Training_vectorized.csv", index=False)

        # dff_test = list(vectorizer.transform(self.test_data["Review"]).toarray())
        # vocab_test = vectorizer.vocabulary_
        # keys_test = list(vocab_test.keys())
        # dff_test_df = pd.DataFrame(dff_test, columns=keys_test)
        # dff_test_df.reset_index(drop=True, inplace=True)
        # self.test_data.reset_index(drop=True, inplace=True)
        # Final_Test = pd.concat([self.test_data, dff_test_df], axis=1)
        # Final_Test.drop(["Review"], axis=1, inplace=True)
        # Final_Test.to_csv("Final_Test_vectorized", index=False)

        # save the vectorizer to disk
        joblib.dump(vectorizer, "vectorizer.pkl")
        return Final_Training_data


if __name__ == "__main__":
    # data_utils = DataUtils()
    # jigsaw_data, c3_data, ruddit_data = data_utils.load_different_data()
    # Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
    # data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
    # data = data.sample(5000)
    # # data.reset_index(inplace=True)

    # data_process = DataProcess(data, jigsaw_data, c3_data, ruddit_data)
    # data_process.check_null_values()
    # data_processed = data_process.apply_all_processing_on_train_test_data()
    # # vectrorize = Vectorization(data)
    # # Final_Training_data = vectrorize.vectorize()
    # print(data.shape)
    # print(data.head())
    # fe = FeatureEngineering(data)
    # data = fe.add_features()
    # print(data.shape)
    # print(data.head())
    pass

