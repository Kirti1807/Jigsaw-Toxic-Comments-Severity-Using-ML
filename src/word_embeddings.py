import numpy as np
import pandas as pd
from gensim.models import FastText
from application_logger import CustomApplicationLogger
import re
import nltk

from data_ingestion import DataUtils
from data_processing import DataProcess

stop_words = nltk.corpus.stopwords.words("english")


class WordEmbeddings:
    def __init__(self) -> None:
        self.logger = CustomApplicationLogger()
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\FeatureEngineeringLogs.txt",
            "a+",
        )

    def averaged_word2vec_vectorizer(self, corpus, model, num_features):
        self.logger.log(self.file_object, "In averaged_word2vec_vectorizer")
        vocabulary = set(model.wv.index2word)

        def average_word_vectors(words, model, vocabulary, num_features):
            feature_vector = np.zeros((num_features,), dtype="float64")
            nwords = 0.0

            for word in words:
                if word in vocabulary:
                    nwords = nwords + 1.0
                    feature_vector = np.add(feature_vector, model.wv[word])
            if nwords:
                feature_vector = np.divide(feature_vector, nwords)

            return feature_vector

        features = [
            average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
            for tokenized_sentence in corpus
        ]
        self.logger.log(self.file_object, "Completed: averaged_word2vec_vectorizer")
        return np.array(features)

    def get_word_embeddings(self, tokenized_docs):
        model = FastText.load(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\Saved_models\model.model"
        )
        doc_vecs_ft = self.averaged_word2vec_vectorizer(tokenized_docs, model, 300)
        return doc_vecs_ft

    def normalize_document(self, doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r"[^a-zA-Z0-9\s]", "", doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        # tokenize document
        tokens = nltk.word_tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        doc = " ".join(filtered_tokens)
        return doc

    def get_word_embeddings_for_training_data(self):
        try:
            self.logger.log(
                self.file_object, "In get_word_embeddings_for_training_data"
            )
            data_utils = DataUtils()
            Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
            data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
            data_process = DataProcess(data)
            data_process.check_null_values()
            data_processed = data_process.apply_all_processing_on_train_test_data()
            corpus = data_processed["text"]
            corpus = corpus.values
            corpus = [self.normalize_document(doc) for doc in corpus]
            tokenized_docs = [doc.split() for doc in corpus]
            docs_vecs_ft = self.get_word_embeddings(corpus)
            docs_vecs_ft_df = pd.DataFrame(docs_vecs_ft)
            docs_vecs_ft_df.to_csv(
                r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\processed_data\docs_vecs_ft.csv"
            )
            self.logger.log(
                self.file_object, "Completed get_word_embeddings_for_training_data"
            )
            return docs_vecs_ft_df
        except Exception as e:
            self.logger.log(
                self.file_object,
                "Error while getting word embeddings for training data",
            )
            self.logger.log(self.file_object, str(e))
            raise Exception()

    def load_word_embeddings(self): 
        try: 
            self.logger.log(self.file_object, "In load_word_embeddings") 
            docs_vecs_ft_df = pd.read_csv( 
                r"E:\QnAMedical\Jigsaw Text Comment Severity\dataset\processed_data\docs_vecs_ft.csv"  
            ) 
            return docs_vecs_ft_df 
        except Exception as e: 
            self.logger.log(self.file_object, "Error while loading word embeddings") 
            self.logger.log(self.file_object, str(e)) 
            raise Exception() 

if __name__ == "__main__":
    # prediction = WordEmbeddings(fgvh)
    # text = ["Great article that unions hate to hear. Bottom line is Canada needs workers. There will always be that 5 of the workforce that beat the system or won't work."]
    # doc_vecs_ft = prediction.get_word_embeddings(text)
    # print(doc_vecs_ft.shape)
    # word_embeds = WordEmbeddings()
    # docs_vecs_ft_df = word_embeds.get_word_embeddings_for_training_data() 
    pass   
