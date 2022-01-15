import numpy as np

# import sklearn regression evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from application_logger import CustomApplicationLogger


class Evaluation:
    def __init__(self) -> None:
        self.logger = CustomApplicationLogger()
        self.file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\BestModelFinderLogs.txt", "a+"
        )

    def mean_absolute_percentage_error(self, y_true, y_pred):
        self.logger.log(
            self.file_object,
            "Entered the mean_absolute_percentage_error method of the Evaluation class",
        )
        self.logger.log(
            self.file_object,
            "The mean absolute percentage error value is: "
            + str(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        )
        self.logger.log(
            self.file_object,
            "Exited the mean_absolute_percentage_error method of the Evaluation class",
        )
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def mean_squared_error(self, y_true, y_pred):
        try:
            self.logger.log(
                self.file_object,
                "Entered the mean_squared_error method of the Evaluation class",
            )
            mse = mean_squared_error(y_true, y_pred)
            self.logger.log(
                self.file_object, "The mean squared error value is: " + str(mse),
            )

            return mse
        except Exception as e:
            self.logger.log(
                self.file_object,
                "Exception occured in mean_squared_error method of the Evaluation class. Exception message:  "
                + str(e),
            )
            self.logger.log(
                self.file_object,
                "Exited the mean_squared_error method of the Evaluation class",
            )
            raise Exception()

    def r2_score(self, y_true, y_pred):
        try:
            self.logger.log(
                self.file_object, "Entered the r2_score method of the Evaluation class",
            )
            r2 = r2_score(y_true, y_pred)
            self.logger.log(
                self.file_object, "The r2 score value is: " + str(r2),
            )
            self.logger.log(
                self.file_object, "Exited the r2_score method of the Evaluation class",
            )
            return r2
        except Exception as e:
            self.logger.log(
                self.file_object,
                "Exception occured in r2_score method of the Evaluation class. Exception message:  "
                + str(e),
            )
            self.logger.log(
                self.file_object, "Exited the r2_score method of the Evaluation class",
            )
            raise Exception()

    def root_mean_squared_error(self, y_true, y_pred):
        try:
            self.logger.log(
                self.file_object,
                "Entered the root_mean_squared_error method of the Evaluation class",
            )
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            self.logger.log(
                self.file_object, "The root mean squared error value is: " + str(rmse),
            )
            return rmse
        except Exception as e:
            self.logger.log(
                self.file_object,
                "Exception occured in root_mean_squared_error method of the Evaluation class. Exception message:  "
                + str(e),
            )
            self.logger.log(
                self.file_object,
                "Exited the root_mean_squared_error method of the Evaluation class",
            )
            raise Exception()


class ModelEvaluater:
    def __init__(self, x_test, y_test) -> None:
        self.x_test = x_test
        self.y_test = y_test
        self.logger = CustomApplicationLogger()
        self.file_object = open(
            r"E:\QnAMedical\Credit Card Fraud\logs\BestModelFinderLogs.txt", "a+"
        )
        self.evaluator = Evaluation()

    def evaluate_trained_models(self, lg_model, rf_model, lgbm_model, xgb_model):
        try:
            self.logger.log(
                self.file_object,
                "Entered the evaluate_trained_models method of the ModelEvaluater class",
            )
            lg_pred = lg_model.predict(self.x_test)
            rf_pred = rf_model.predict(self.x_test)
            lgbm_pred = lgbm_model.predict(self.x_test)
            xgb_pred = xgb_model.predict(self.x_test)
            self.logger.log(
                self.file_object,
                "The mean absolute percentage error value for the CatBoost model is: "
                + str(
                    self.evaluator.mean_absolute_percentage_error(self.y_test, lg_pred)
                ),
            )
            self.logger.log(
                self.file_object,
                "The mean absolute percentage error value for the Random Forest model is: "
                + str(
                    self.evaluator.mean_absolute_percentage_error(self.y_test, rf_pred)
                ),
            )
            self.logger.log(
                self.file_object,
                "The mean absolute percentage error value for the Light GBM model is: "
                + str(
                    self.evaluator.mean_absolute_percentage_error(
                        self.y_test, lgbm_pred
                    )
                ),
            )
            self.logger.log(
                self.file_object,
                "The mean absolute percentage error value for the XGBoost model is: "
                + str(
                    self.evaluator.mean_absolute_percentage_error(self.y_test, xgb_pred)
                ),
            )
            self.logger.log(
                self.file_object,
                "MSE for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
                    self.evaluator.mean_squared_error(self.y_test, lg_pred),
                    self.evaluator.mean_squared_error(self.y_test, rf_pred),
                    self.evaluator.mean_squared_error(self.y_test, lgbm_pred),
                    self.evaluator.mean_squared_error(self.y_test, xgb_pred),
                ),
            )

            self.logger.log(
                self.file_object,
                self.file_object,
                "RMSE for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
                    self.evaluator.root_mean_squared_error(self.y_test, lg_pred),
                    self.evaluator.root_mean_squared_error(self.y_test, rf_pred),
                    self.evaluator.root_mean_squared_error(self.y_test, lgbm_pred),
                    self.evaluator.root_mean_squared_error(self.y_test, xgb_pred),
                ),
            )

            self.logger.log(
                self.file_object,
                self.file_object,
                "R2 for catboost {}, random forest {}, light gbm {}, xgboost {}".format(
                    self.evaluator.r2_score(self.y_test, lg_pred),
                    self.evaluator.r2_score(self.y_test, rf_pred),
                    self.evaluator.r2_score(self.y_test, lgbm_pred),
                    self.evaluator.r2_score(self.y_test, xgb_pred),
                ),
            )

            self.logger.log(
                self.file_object,
                "Exited the evaluate_trained_models method of the ModelEvaluater class",
            )
        except Exception as e:
            self.logger.log(
                self.file_object,
                "Exception occured in evaluate_trained_models method of the ModelEvaluater class. Exception message:  "
                + str(e),
            )
            self.logger.log(
                self.file_object,
                "Exited the evaluate_trained_models method of the ModelEvaluater class",
            )
            raise Exception()

