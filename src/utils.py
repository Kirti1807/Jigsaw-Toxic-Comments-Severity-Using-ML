import pickle
import os
import shutil
from application_logger import CustomApplicationLogger


class File_Ops:
    def __init__(self, file_object, logger) -> None:
        self.file_object = file_object
        self.logger = logger
        self.model_directory = r"E:\QnAMedical\Jigsaw Text Comment Severity\saved_model"

    def save_models(self, model, filename):
        try:
            path = os.path.join(
                self.model_directory, filename
            )  # create seperate directory for each cluster
            if os.path.isdir(
                path
            ):  # remove previously existing models for each clusters
                shutil.rmtree(self.model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + "/" + filename + ".pkl", "wb") as f:
                pickle.dump(model, f)  # save the model to file
            # self.logger_object.logger(
            #     self.file_object, "Model File " + filename + " saved."
            # )

        except Exception as e:
            self.logger.logger(
                self.file_object,
                "Exception occured in save_model method of the Model_Finder class. Exception message:  "
                + str(e),
            )
            raise Exception()

    def load_models(self, filename):
        try:
            with open(
                self.model_directory + filename + "/" + filename + ".pkl", "rb"
            ) as f:
                self.logger.logger(
                    self.file_object, "Model File " + filename + " loaded."
                )
                return pickle.load(f)
        except Exception as e:
            self.logger.logger(
                self.file_object,
                "Exception occured in load_model method of the Model_Finder class. Exception message:  "
                + str(e),
            )
            raise Exception()

    def get_correct_models(self, cluster_number):
        self.logger.logger(
            self.file_object, "Entered file_loader",
        )
        try:
            self.cluster_number = cluster_number
            self.folder_name = self.model_directory
            self.list_of_model_files = []
            self.list_of_files = os.listdir(self.folder_name)
            for self.file in self.list_of_files:
                try:
                    if self.file.index(str(self.cluster_number)) != -1:
                        self.model_name = self.file
                except:
                    continue
            self.model_name = self.model_name.split(".")[0]
            self.logger.logger(
                self.file_object, "Success find correct model file.",
            )
            return self.model_name
        except Exception as e:
            self.logger.logger(
                self.file_object, "Error got:-  " + str(e),
            )
            self.logger.logger(
                self.file_object,
                "Exited the find_correct_model_file method with Failure",
            )
            raise Exception()

