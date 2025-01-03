import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.BaseRecommender import BaseRecommender
from Evaluation.Evaluator import EvaluatorHoldout
import gc
import optuna
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
#'topK': None, 'l2_norm': 54.002102725300844, 'normalize_matrix': False}. 
# Best is trial 34 with value: 0.029328203513068373.

class ScoresHybridRecommender(BaseRecommender):
    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2, recommender_3, recommender_4):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train, dtype=np.bool_)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4

    def fit(self, alpha=0.5, beta=0.5, gamma=0, theta=0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array)
        item_weights = (item_weights_1 * self.alpha +
                        item_weights_2 * self.beta +
                        item_weights_3 * self.gamma +
                        item_weights_4 * self.theta)
        return item_weights


# Carica il file CSV -> URM
data = pd.read_csv('data_train.csv')

# Trova il massimo user_id e item_id per dimensionare correttamente la matrice
max_user_id = data['user_id'].max()
max_item_id = data['item_id'].max()

# Crea la matrice sparsa con tipo booleano
URM_all = csr_matrix((data['data'].astype(np.bool_), 
                      (data['user_id'], data['item_id'])), 
                      shape=(max_user_id + 1, max_item_id + 1))

# Divisione del dataset
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
#URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.8)

# Valutatori
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])
#evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

# Carica il file CSV -> ICM
data_ICM = pd.read_csv('data_ICM_metadata.csv')

# Estrai le colonne necessarie
item_ids = data_ICM['item_id'].values
feature_ids = data_ICM['feature_id'].values
data_values = data_ICM['data'].values

# Trova il numero massimo di item_id e feature_id per determinare la dimensione della matrice
num_items = item_ids.max() + 1
num_features = feature_ids.max() + 1

# Crea la matrice ICM come matrice sparsa con tipo booleano
ICM_matrix = sps.csr_matrix((data_values.astype(np.bool_), 
                             (item_ids, feature_ids)), 
                             shape=(num_items, num_features))

# Verifica se ICM_matrix ha lo stesso numero di righe di URM_train
if ICM_matrix.shape[0] < URM_train.shape[1]:
    ICM_matrix = sps.vstack([ICM_matrix, sps.csr_matrix((URM_train.shape[1] - ICM_matrix.shape[0], ICM_matrix.shape[1]), dtype=np.bool_)])

# Concatenazione di URM e ICM
stacked_URM = sps.vstack([URM_train, ICM_matrix.T], format='csr', dtype=np.bool_)

# Ottimizzazione con Optuna
class SaveResults:
    def __init__(self, save_path="results.csv"):
        self.save_path = save_path
        self.results_df = pd.DataFrame(columns=["result"])

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]

        self.results_df = self.results_df._append(hyperparam_dict, ignore_index=True)
        self.results_df.to_csv(self.save_path, index=False)


def objective_function(optuna_trial):

    from Recommenders.FactorizationMachines.LightFMRecommender import LightFMCFRecommender
    recommender_instance = LightFMCFRecommender(URM_train)
    # 0.030721241087486122 and parameters: {'epochs': 300, 
    # 'n_components': 176, 'loss': 'warp-kos', 'sgd_mode': 'adadelta', 
    # 'learning_rate': 0.002906763130181892, 'item_alpha': 0.0001284138694547123, 
    # 'user_alpha': 1.8789357078260402e-05}
    recommender_instance.fit(
    epochs = optuna_trial.suggest_categorical("epochs", [300]),
    n_components = optuna_trial.suggest_int("n_components", 1, 200),
    loss = optuna_trial.suggest_categorical("loss", ['bpr', 'warp', 'warp-kos']),
    sgd_mode = optuna_trial.suggest_categorical("sgd_mode", ['adagrad', 'adadelta']),
    learning_rate = optuna_trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
    item_alpha = optuna_trial.suggest_float("item_alpha", 1e-5, 1e-2, log=True),
    user_alpha = optuna_trial.suggest_float("user_alpha", 1e-5, 1e-2, log=True),
    
    )

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    # Forza la garbage collection
    del recommender_instance
    gc.collect()

    return result_df.loc[10]["MAP"]


class SaveResults(object):
    
    def __init__(self):
        self.results_df = pd.DataFrame(columns = ["result"])
    
    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        
        self.results_df = self.results_df._append(hyperparam_dict, ignore_index=True)


optuna_study = optuna.create_study(direction="maximize")
        
save_results = SaveResults()
        
optuna_study.optimize(objective_function,
                      callbacks=[save_results],
                      n_trials = 500)

print(optuna_study.best_trial.params)

