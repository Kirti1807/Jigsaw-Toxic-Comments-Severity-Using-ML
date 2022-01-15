import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from application_logger import CustomApplicationLogger
from utils import File_Ops


class KMeansClustering:
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def elbow_plot(self, data):
        self.logger_object.log(
            self.file_object,
            "Entered the elbow_plot method of the KMeansClustering class",
        )
        wcss = []  # initializing an empty list
        try:
            for i in range(1, 11):
                kmeans = KMeans(
                    n_clusters=i, init="k-means++", random_state=42
                )  # initializing the KMeans object
                kmeans.fit(data)  # fitting the data to the KMeans Algorithm
                wcss.append(kmeans.inertia_)
            plt.plot(
                range(1, 11), wcss
            )  # creating the graph between WCSS and the number of clusters
            plt.title("The Elbow Method")
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            # plt.show()
            plt.savefig(
                r"E:\QnAMedical\Jigsaw Text Comment Severity\saved_model\K-Means_Elbow.PNG"
            )  # saving the elbow plot locally
            # finding the value of the optimum cluster programmatically
            self.kn = KneeLocator(
                range(1, 11), wcss, curve="convex", direction="decreasing"
            )
            self.logger_object.log(
                self.file_object,
                "The optimum number of clusters is: " + str(self.kn.knee),
            )
            return self.kn.knee

        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in elbow_plot method of the KMeansClustering class. Exception message:  "
                + str(e),
            )
            self.logger_object.log(
                self.file_object,
                "Finding the number of clusters failed. Exited the elbow_plot method of the KMeansClustering class",
            )
            raise Exception()

    def create_clusters(self, data, number_of_clusters):
        self.logger_object.log(
            self.file_object,
            "Entered the create_clusters method of the KMeansClustering class",
        )
        self.data = data
        try:
            self.kmeans = KMeans(
                n_clusters=number_of_clusters, init="k-means++", random_state=42
            )
            # self.data = self.data[~self.data.isin([np.nan, np.inf, -np.inf]).any(1)]
            self.y_kmeans = self.kmeans.fit_predict(data)  #  divide data into clusters

            self.file_op = File_Ops(self.file_object, self.logger_object)
            self.save_model = self.file_op.save_models(
                self.kmeans, "KMeans"
            )  # saving the KMeans model to directory
            # passing 'Model' as the functions need three parameters

            self.data[
                "Cluster"
            ] = (
                self.y_kmeans
            )  # create a new column in dataset for storing the cluster information
            self.logger_object.log(
                self.file_object,
                "succesfully created " + str(self.kn.knee) + "clusters.",
            )
            return self.data
        except Exception as e:
            self.logger_object.log(
                self.file_object,
                "Exception occured in create_clusters method of the KMeansClustering class. Exception message:  "
                + str(e),
            )
            raise Exception()

    # def apply_clustering(self, docs):
    #     try:
    #         number_of_clusters = self..elbow_plot(X_train)

    #         # Divide the data into clusters
    #         X_train = self..create_clusters(docs, number_of_clusters)
