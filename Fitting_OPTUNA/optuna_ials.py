import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import matplotlib.pyplot as plt
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender


from Recommenders.BaseRecommender import BaseRecommender

class ScoresHybridRecommender(BaseRecommender):
   

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2,recommender_3,recommender_4):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        self.recommender_3 = recommender_3
        self.recommender_4 = recommender_4
    def fit(self, alpha = 0.5, beta=0.5, gamma=0, theta=0.2):
        self.alpha = alpha      
        self.beta = beta
        self.gamma= gamma
        self.theta= theta
    def _compute_item_score(self, user_id_array, items_to_compute):
        
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights_3 = self.recommender_3._compute_item_score(user_id_array)
        item_weights_4 = self.recommender_4._compute_item_score(user_id_array)
        item_weights = item_weights_1*self.alpha + item_weights_2*(self.beta) +item_weights_3*(self.gamma) +item_weights_3*(self.theta)

        return item_weights



# Carica il file CSV  -> URM
data = pd.read_csv('data_train.csv')

# Trova il massimo user_id e item_id per dimensionare correttamente la matrice
max_user_id = data['user_id'].max()
max_item_id = data['item_id'].max()

# Crea la matrice sparsa
URM_all = csr_matrix((data['data'], (data['user_id'], data['item_id'])), shape=(max_user_id + 1, max_item_id + 1))
from Evaluation.Evaluator import EvaluatorHoldout

URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])



# Print initial URM_train matrix
print("URM_train (before concatenation):")
print(URM_train)


# Carica il file CSV -> ICM
data_ICM = pd.read_csv('data_ICM_metadata.csv')

# Estrai le colonne necessarie
item_ids = data_ICM['item_id'].values
feature_ids = data_ICM['feature_id'].values
data_values = data_ICM['data'].values

# Trova il numero massimo di item_id e feature_id per determinare la dimensione della matrice
num_items = item_ids.max() + 1
num_features = feature_ids.max() + 1

# Crea la matrice ICM come matrice sparsa
ICM_matrix = sps.csr_matrix((data_values, (item_ids, feature_ids)), shape=(num_items, num_features))

# Print initial ICM matrix
print("\nICM_matrix (before concatenation):")
print(ICM_matrix)

# Verifica se ICM_matrix ha lo stesso numero di righe di URM_train, altrimenti riempie le righe mancanti
if ICM_matrix.shape[0] < URM_train.shape[1]:
    ICM_matrix = sps.vstack([ICM_matrix, sps.csr_matrix((URM_train.shape[1] - ICM_matrix.shape[0], ICM_matrix.shape[1]))])

# Concatenate URM and ICM
stacked_URM = sps.vstack([URM_train, ICM_matrix.T])
stacked_URM = sps.csr_matrix(stacked_URM)

# Print the concatenated stacked_URM matrix
print("\nstacked_URM (after concatenation):")
print(stacked_URM)

import optuna
import pandas as pd
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

def objective_function(optuna_trial):

    from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
    recommender_instance = IALSRecommender(URM_train)
    recommender_instance.fit(
    epochs=optuna_trial.suggest_int("epochs", 100, 500),  # Intervallo aumentato per considerare un numero maggiore di epoche
    num_factors=optuna_trial.suggest_int("num_factors", 10, 200),  # Variabilità maggiore per il numero di fattori
    confidence_scaling=optuna_trial.suggest_categorical("confidence_scaling", ["linear", "log", "exp"]),  # Aggiunta di altre opzioni
    alpha=optuna_trial.suggest_loguniform("alpha", 1e-3, 10),  # Espanso per esplorare un ampio intervallo logaritmico
    epsilon=optuna_trial.suggest_loguniform("epsilon", 1e-5, 1e-1),  # Valori logaritmici più ampi per epsilon
    reg=optuna_trial.suggest_loguniform("reg", 1e-5, 1e-2),  # Penalizzazione, con un intervallo logaritmico
    init_mean=optuna_trial.suggest_uniform("init_mean", -0.1, 0.1),  # Inizializzazione uniforme più stretta
    init_std=optuna_trial.suggest_uniform("init_std", 0.05, 0.2),  # Deviazione standard per la normalizzazione

)

    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
    
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


"""
trace_users = pd.read_csv('data_target_users_test.csv')
trace_users = trace_users.to_numpy()
# Open the file for writing recommendations
with open('sample_submission.csv', 'w') as file:
    
    file.write("user_id,item_list\n")
    # Loop over each user
    for user_id in trace_users:
        # Get the top 10 recommendations for the user
        recommended_items = recommender_ItemKNNCFCBF.recommend(user_id_array=user_id, cutoff=10)
        stringa=  (' '.join(map(str, np.array(recommended_items))))
        file.write(f"{user_id[0]}, {stringa[1: len(stringa) -1]}\n")
        
print("Recommendations have been written to 'sample_submission.csv'")
"""