import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
import gc
import optuna
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.BaseRecommender import BaseRecommender
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender


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

# Trova gli item comuni tra URM_train e ICM_matrix
common_items = np.intersect1d(np.arange(URM_train.shape[1]), np.arange(max_item_id + 1))

# Crea la matrice ICM filtrata per includere solo gli item comuni
ICM_matrix = csr_matrix((data['data'].astype(np.bool_), 
                         (data['user_id'], data['item_id'])), 
                         shape=(max_user_id + 1, max_item_id + 1))

# Assicurati di non accedere a indici fuori dalla dimensione
common_items = np.intersect1d(np.arange(ICM_matrix.shape[1]), common_items)

ICM_matrix = ICM_matrix[:, common_items]  # Mantieni solo gli item comuni
URM_train = URM_train[:, common_items]    # Mantieni solo gli item comuni

# Valutatori
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])


# Ottimizzazione con Optuna
class SaveResults:
    def __init__(self, save_path="results.csv"):
        self.save_path = save_path
        self.results_df = pd.DataFrame(columns=["result"])

    def __call__(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]
        self.results_df = self.results_df.append(hyperparam_dict, ignore_index=True)
        self.results_df.to_csv(self.save_path, index=False)

def objective_function(optuna_trial):

    # Ottieni i parametri da Optuna
    alpha = optuna_trial.suggest_int("alpha_features", 0, 30)
    # Stampa dimensioni iniziali
    print(f"Dimensioni iniziali della matrice ICM: {ICM_matrix.shape}")

    # Fase 1: Filtra le feature che non hanno item associati
    feature_sums = ICM_matrix.sum(axis=0).A1  # Somma per colonna
    non_empty_features = np.where(feature_sums > alpha)[0]
    ICM_matrix_filtered = ICM_matrix[:, non_empty_features]

    # Crea e allena il raccomandatore
    recommender_instance = ItemKNNCBFRecommender(ICM_train=ICM_matrix_filtered, URM_train=URM_train)
    recommender_instance.fit(
        topK=optuna_trial.suggest_int("topK", 5, 1000),
        shrink=optuna_trial.suggest_int("shrink", 0, 100),
        similarity="cosine",
        normalize=optuna_trial.suggest_categorical("normalize", [True, False]),
        feature_weighting=optuna_trial.suggest_categorical("feature_weighting", ["BM25", "TF-IDF", "none"]),
    )

    # Valutazione
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)

    # Forza la garbage collection
    del recommender_instance
    gc.collect()

    # Ritorna la MAP
    return result_df.loc[10]["MAP"]

# Ottimizzazione con Optuna
optuna_study = optuna.create_study(direction="maximize")

# Salvataggio dei risultati
save_results = SaveResults()

# Esegui l'ottimizzazione
optuna_study.optimize(objective_function,
                      callbacks=[save_results],
                      n_trials=500)

# Stampa i migliori parametri
print(optuna_study.best_trial.params)