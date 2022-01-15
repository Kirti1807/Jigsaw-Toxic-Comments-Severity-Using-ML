from joblib.externals.loky.backend.spawn import import_main_path
from numpy.lib.function_base import gradient
import pandas as pd
import numpy as np
import logging

from scipy.sparse import data
from scipy.sparse.construct import random

# from validate_data import DatavalidationTest

# import GrideSearchCV
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBRegressor
from data_ingestion import DataUtils
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import optuna
from sklearn import linear_model

# from evaluate_models import Evaluation
# from utils import Submission
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import catboost
from data_processing import DataProcess, DataValidation
from feature_engineering import Vectorization
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from application_logger import CustomApplicationLogger

# from pystacknet.pystacknet import StackNetRegressor


class Hyperparameters_Optimization:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_decisiontrees(self, trial):
        # criterion = trial.suggest_categorical("criterion", ("squared_error", "friedman_mse", "absolute_error", "poisson"))
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split,
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def optimize_randomforest(self, trial):
        logging.info("optimize_randomforest")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def Optimize_Adaboost_regressor(self, trial):
        logging.info("Optimize_Adaboost_regressor")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def Optimize_LightGBM(self, trial):
        logging.info("Optimize_LightGBM")
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = LGBMRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def Optimize_Xgboost_regressor(self, trial):
        logging.info("Optimize_Xgboost_regressor")
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = xgb.XGBRegressor(**param)
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy

    def Optimize_Catboost_regressor(self, trial):
        logging.info("Optimize_Catboost_regressor")
        param = {
            "iterations": trial.suggest_int("iterations", 1, 200),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-7, 1.0),
            "depth": trial.suggest_int("depth", 1, 16),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-7, 10.0),
            "border_count": trial.suggest_int("border_count", 1, 20),
            "rsm": trial.suggest_uniform("rsm", 0.5, 1.0),
            "od_type": trial.suggest_categorical(
                "od_type", ("IncToDec", "Iter", "None")
            ),
            "od_wait": trial.suggest_int("od_wait", 1, 20),
            "random_seed": trial.suggest_int("random_seed", 1, 20),
            "loss_function": trial.suggest_categorical(
                "loss_function", ("RMSE", "MAE")
            ),
        }
        reg = CatBoostRegressor(**param)
        reg.fit(self.x_train, self.y_train)
        val_accuracy = reg.score(self.x_test, self.y_test)
        return val_accuracy


class ModelTraining:
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def decision_trees(self, fine_tuning=True):
        logging.info("Entered for training Decision Trees model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_decisiontrees, n_trials=100)
                trial = study.best_trial
                # criterion = trial.params["criterion"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                reg = DecisionTreeRegressor(
                    max_depth=max_depth, min_samples_split=min_samples_split,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = DecisionTreeRegressor(
                    criterion="squared_error", max_depth=7, min_samples_split=13
                )

                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Decision Trees model")
            logging.error(e)
            return None

    def random_forest(self, fine_tuning=True):
        logging.info("Entered for training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                print("Best parameters : ", trial.params)
                reg = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = RandomForestRegressor(
                    n_estimators=152, max_depth=20, min_samples_split=17
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def adabooost_regressor(self, fine_tuning=True):
        logging.info("Entered for training Adaboost regressor model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.Optimize_Adaboost_regressor, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                reg = AdaBoostRegressor(
                    n_estimators=n_estimators, learning_rate=learning_rate
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = AdaBoostRegressor(n_estimators=200, learning_rate=0.01)
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Adaboost regressor model")
            logging.error(e)
            return None

    def LightGBM(self, fine_tuning=True):
        logging.info("Entered for training LightGBM model")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.Optimize_LightGBM, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                learning_rate = trial.params["learning_rate"]
                reg = LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model")
            logging.error(e)
            return None

    def xgboost(self, fine_tuning=True):
        logging.info("Entered for training XGBoost model")
        try:
            if fine_tuning:
                hy_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.Optimize_Xgboost_regressor, n_trials=100)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                learning_rate = trial.params["learning_rate"]
                max_depth = trial.params["max_depth"]
                reg = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                )
                reg.fit(self.x_train, self.y_train)
                return reg

            else:
                model = xgb.XGBRegressor(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training XGBoost model")
            logging.error(e)
            return None

    def Catboost(self, fine_tuning=True):
        logging.info("Entered for training Catboost model")
        try:
            if fine_tuning:
                hy_opt = Hyperparameters_Optimization(
                    self.x_train, self.y_train, self.x_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.Optimize_Catboost_regressor, n_trials=100)
                trial = study.best_trial
                iterations = trial.params["iterations"]
                depth = trial.params["depth"]
                l2_leaf_reg = trial.params["l2_leaf_reg"]
                learning_rate = trial.params["learning_rate"]
                logging.info("Best parameters : ", trial.params)
                reg = CatBoostRegressor(
                    iterations=iterations,
                    depth=depth,
                    l2_leaf_reg=l2_leaf_reg,
                    learning_rate=learning_rate,
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = CatBoostRegressor(
                    iterations=200, depth=16, l2_leaf_reg=0.01, learning_rate=0.01
                )
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Catboost model")
            logging.error(e)
            return None

    def stacking_regression(self):
        logging.info("Entered for stacking model")
        try:
            decision_tree = DecisionTreeRegressor(max_depth=6, min_samples_split=7)
            rf_tree = RandomForestRegressor(
                n_estimators=152, max_depth=20, min_samples_split=17
            )
            adaboost = AdaBoostRegressor(n_estimators=200, learning_rate=0.01)
            xgb_reg = XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=20)
            # cat_reg = CatBoostRegressor(
            #     iterations=200, depth=20, l2_leaf_reg=0.01, learning_rate=1e-7
            # )
            # lr = LinearRegression() # use random forest here
            reg = StackingRegressor(
                regressors=[decision_tree, rf_tree, adaboost, xgb_reg],
                meta_regressor=rf_tree,
            )
            reg.fit(self.x_train, self.y_train)
            return reg
        except Exception as e:
            logging.error("Error in stacking model")
            logging.error(e)
            return None


class BestModelFinder:
    def __init__(self) -> None:
        self.file_object = open(
            r"E:\QnAMedical\Jigsaw Text Comment Severity\logs\BestModelFinderLogs.txt",
            "a+",
        )
        self.logger = CustomApplicationLogger()

    def train_all_models(self, x_train, x_test, y_train, y_test):
        try:
            # Logisitc Regression Model
            model_train = ModelTraining(x_train, x_test, y_train, y_test)
            lg_model = model_train.Catboost(fine_tuning=False)

            if lg_model is not None:
                self.logger.logger(
                    self.file_object, "Catboost model is trained successfully",
                )

            # decision_trees
            # model_train = ModelTraining(x_train, x_test, y_train, y_test)
            # dt_model = model_train.decision_trees(fine_tuning=False)
            # if dt_model is not None:
            #     self.logger.logger(
            #         self.file_object, "Decision Tree model is trained successfully",
            #     )
            # Random Forest

            model_train = ModelTraining(x_train, x_test, y_train, y_test)
            rf_model = model_train.random_forest(fine_tuning=False)
            if rf_model is not None:
                self.logger.logger(
                    self.file_object, "Random Forest model is trained successfully",
                )
            # LightGBM

            model_train = ModelTraining(x_train, x_test, y_train, y_test)
            lgbm_model = model_train.LightGBM(fine_tuning=True)
            if lgbm_model is not None:
                self.logger.logger(
                    self.file_object, "LightGBM model is trained successfully",
                )

            # XGBoost

            model_train = ModelTraining(x_train, x_test, y_train, y_test)
            xgb_model = model_train.xgboost(fine_tuning=False)
            if xgb_model is not None:
                self.logger.logger(
                    self.file_object, "XGBoost model is trained successfully",
                )

            # print the best model among these by comparing the score of all models

            return lg_model, rf_model, lgbm_model, xgb_model
        except Exception as e:
            # self.logger.logger(self.file_object, str(e))
            raise e


if __name__ == "__main__":
    # data_utils = DataUtils()
    # jigsaw_data, c3_data, ruddit_data = data_utils.load_different_data()
    # Jigsaw, c3_data, ruddit_data = data_utils.prepare_data()
    # data = data_utils.concatenate_dfs(Jigsaw, c3_data, ruddit_data)
    # data = data.sample(100)
    # data_process = DataProcess(data, jigsaw_data, c3_data, ruddit_data)
    # data_process.check_null_values()
    # data_processed = data_process.apply_all_processing_on_train_test_data()
    # vectrorize = Vectorization(data)
    # Final_Training_data = vectrorize.vectorize()
    # data_val = DataValidation(Final_Training_data)
    # x_train, x_test, y_train, y_test = data_val.data_splitting()
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # model_train = ModelTraining(x_train, y_train, x_test, y_test)
    # # ran_for = model_train.random_forest(fine_tuning=True)
    # # ada_for = model_train.adabooost_regressor(fine_tuning=True)
    # # lgbm_for = model_train.LightGBM(fine_tuning=True)
    # # xgb_for = model_train.xgboost(fine_tuning=True)
    # # catboost_for = model_train.Catboost(fine_tuning=True)
    # stack_for = model_train.stacking_regression()
    # print(stack_for)
    pass

