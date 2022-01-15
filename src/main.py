import numpy as np
from application_logger import CustomApplicationLogger
from evaluation import *
from model_dev import ModelTraining, BestModelFinder
from data_ingestion import  *
from eda_src import EDA
from data_processing import DataProcess
from clustering import KMeansClustering
from utils import File_Ops
from sklearn.model_selection import train_test_split 
from word_embeddings import WordEmbeddings 

def main(): 
    try:  
        logger = CustomApplicationLogger() 
        file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\MainLogs.txt", "a+"
        )
        logger.log(file_object, "Entered the main method of the script") 
        data_utils = DataUtils()
        Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
        data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
        logger.log(file_object, "DataIngestion Successful") 

        logger.log(file_object, "EDA Started")  
        eda = EDA(data, Jigsaw, c3_data, ruddit_data)
        eda.data_info()
        eda.exploring_distributions()
        eda.word_cloud()
        eda.skeweness_and_kurtosis()
        logger.log(file_object, "EDA Successful") 

        logger.log(file_object, "DataProcessing Started")
        data_process = DataProcess(data)
        data_process.check_null_values()
        data_processed = data_process.apply_all_processing_on_train_test_data()
        logger.log(file_object, "DataProcessing Successful") 

        logger.log(file_object, "Loading the data") 
        word_embeds = WordEmbeddings() 
        docs_vecs_ft_df = word_embeds.load_word_embeddings() 
        docs_vecs_ft_df.drop(columns=["Unnamed: 0"], axis=1,inplace=True)
        logger.log(file_object, "Loading the data successful") 

        logger.log(file_object, "K-Means Clustering Started") 
        kmeans = KMeansClustering(file_object, logger)
        number_of_clusters = kmeans.elbow_plot(docs_vecs_ft_df)
        docs_vecs_ft_df = kmeans.create_clusters(docs_vecs_ft_df, number_of_clusters) 
        Y = data_processed["y"]
        docs_vecs_ft_df["Labels"] = Y  
        logger.log(file_object, "K-Means Clustering Successful") 

        list_of_clusters = docs_vecs_ft_df["Cluster"].unique()
        docs_vecs_ft_df["Cluster"].value_counts()
        for i in list_of_clusters:
            cluster_data = docs_vecs_ft_df[docs_vecs_ft_df["Cluster"] == i]
            cluster_features = cluster_data.drop(["Labels", "Cluster"], axis=1)
            cluster_label = cluster_data["Labels"]

            # x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1 / 3, random_state=355)
            X = docs_vecs_ft_df.drop(["Labels", "Cluster"], axis=1)
            y = docs_vecs_ft_df[["Labels"]]
    
            # use ravel on y
            y = y.values.ravel()
            y.reshape(1, -1)
            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=355
            )
            ModelTrainer = BestModelFinder()
            lg_model, rf_model, lgbm_model, xgb_model = ModelTrainer.train_all_models(
                x_train, y_train, x_test, y_test
            )
            # saving the best model to the directory.
            file_op = File_Ops(file_object, logger)
            lg_save_model = file_op.save_models(lg_model, "lg_model_" + str(i))
            rf_save_model = file_op.save_models(rf_model, "rf_model_" + str(i))
            lgbm_save_model = file_op.save_models(lgbm_model, "lgbm_model_" + str(i))
            xgb_save_model = file_op.save_models(xgb_model, "xgb_model_" + str(i))

            best_model = ModelEvaluater(x_test, y_test)
            best_model = best_model.evaluate_trained_models(
                lg_model, rf_model, lgbm_model, xgb_model
            )
            # logging the successful Training
            logger.log(file_object, "Successful End of Training")
    except Exception as e:
        logger.log(file_object, "Error Occured")
        logger.log(file_object, str(e))
    finally:
        file_object.close()  



