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

    #print("KNNCF FITTING")
    #{'topK': 5, 'shrink': 16, 'normalize': True}
    #RESULT 0.028830094364301003.
   # recommender_instance = ItemKNNCFRecommender(URM_train)


    #{'ICM_weight': 0.5484923216059217, 'topK': 77, 'shrink': 815, 'normalize': True, 'feature_weighting': 'BM25'
    #0.028106832262946156

   #  {'topK': 66, 'shrink': 65, 'normalize': True}. Best is trial 48 with value: 0.02658754663145145
   # recommender_instance = ItemKNNCFRecommender(URM_train)

   #{'alpha': 0.5563786018254051, 'beta': 0.27116158760203846, 'min_rating': 0, 'topK': 17, 'implicit': False, 'normalize_similarity': True}.
   #  0.030661717555312804.
    #from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
    #recommender_instance = RP3betaRecommender(URM_train)
    

    #{'topK': 16, 'alpha': 0.617210828584666, 'min_rating': 0, 'implicit': False, 'normalize_similarity': True}. 
    #value: 0.0303747957847076.
    #from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
    #recommender_instance= P3alphaRecommender(URM_train)
    

    #{'epochs': 358, 'positive_threshold_BPR': 0.28443693875448256,
    #  'train_with_sparse_weights': False, 'symmetric': False, 
    # 'lambda_i': 0.8591759254062605, 'lambda_j': 0.3064679001750765, 
    # 'learning_rate': 0.0007485576327054323, 'topK': 10, 'sgd_mode': 'adagrad', 'gamma': 0.9273125056326278,
    #  'beta_1': 0.76633332007818, 'beta_2': 0.9889793597419134}
    #result= 0.0259
    
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    from Recommenders.SLIM.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
    recommender_instance= SLIM_BPR_Cython(URM_train)

    recommender_instance.fit(
    epochs=optuna_trial.suggest_int("epochs", 50, 500),
    positive_threshold_BPR=optuna_trial.suggest_float("positive_threshold_BPR", 0.0, 1.0),
    train_with_sparse_weights=optuna_trial.suggest_categorical("train_with_sparse_weights", [None, True, False]),
    allow_train_with_sparse_weights=True,  # Questo è fisso, come definito nella funzione
    symmetric=optuna_trial.suggest_categorical("symmetric", [True, False]),
    lambda_i=optuna_trial.suggest_float("lambda_i", 0.0, 1.0),
    lambda_j=optuna_trial.suggest_float("lambda_j", 0.0, 1.0),
    learning_rate=optuna_trial.suggest_float("learning_rate", 1e-7, 1e-2),
    topK=optuna_trial.suggest_int("topK", 5, 1000),
    sgd_mode=optuna_trial.suggest_categorical("sgd_mode", ["adagrad", "adam", "sgd"]),
    gamma=optuna_trial.suggest_float("gamma", 0.9, 1.0),
    beta_1=optuna_trial.suggest_float("beta_1", 0.5, 0.99),
    beta_2=optuna_trial.suggest_float("beta_2", 0.7, 0.999)
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