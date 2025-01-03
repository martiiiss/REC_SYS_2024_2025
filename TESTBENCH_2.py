import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.BaseRecommender import BaseRecommender
from Evaluation.Evaluator import EvaluatorHoldout
import gc
import optuna
from numpy import linalg as LA
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
#'topK': None, 'l2_norm': 54.002102725300844, 'normalize_matrix': False}. 
# Best is trial 34 with value: 0.029328203513068373.

class DifferentLossScoresHybridRecommender(BaseRecommender):
    """ ScoresHybridRecommender
    Hybrid of two prediction scores R = R1/norm*alpha + R2/norm*(1-alpha) where R1 and R2 come from
    algorithms trained on different loss functions.

    """

    RECOMMENDER_NAME = "DifferentLossScoresHybridRecommender"


    def __init__(self, URM_train, recommender_1, recommender_2):
        super(DifferentLossScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
        #self.recommender_3= recommender_3
        
        
    def fit(self, norm, alpha = 0.5, beta=0.5):

        self.alpha = alpha
        #self.beta=beta
        self.norm = norm


    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array,items_to_compute=items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array,items_to_compute=items_to_compute)
        #item_weights_3 = self.recommender_3._compute_item_score(user_id_array,items_to_compute=items_to_compute)
        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        #norm_item_weights_3 = LA.norm(item_weights_3, self.norm)
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1- self.alpha) 
        #+ item_weights_3 / norm_item_weights_3 * (1 - self.alpha -self.beta) 

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
    print("potrebbe essere un problema")
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


# Ottimizzazione con Optuna
def objective_function(optuna_trial):
    # Parametri suggeriti da Optuna
    alpha = optuna_trial.suggest_float("alpha", 0, 1)
    #beta = optuna_trial.suggest_float("beta", 0, 1)
    norm = optuna_trial.suggest_categorical("norm", [1, 2, np.inf, -np.inf])
    
    # Crea l'istanza del raccomandatore ibrido
    recommender_instance = DifferentLossScoresHybridRecommender(
        URM_train, 
        Last_model,
        Slim_Elasticnet
    )
    
    # Applica il fitting (solo per alpha e norm)
    recommender_instance.fit(alpha=alpha, norm=norm)
    
    # Valutazione
    result_df, _ = evaluator_validation.evaluateRecommender(recommender_instance)
    
    # Garbage collection per evitare memory leak
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

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender


Slim_Elasticnet= SLIMElasticNetRecommender(URM_train)
Slim_Elasticnet.fit(l1_ratio= 0.174358035009766, topK= 631, alpha= 0.00010361443114176747)
#l1_ratio= 0.7508019942429333, alpha= 4.462842551622246e-05, topK= 197)
#l1_ratio= 0.011453254111306543, topK= 422, alpha= 0.0011524047751432808)
#l1_ratio=0.7042629316804924, alpha=4.364330124793646e-05, topK=237)


from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
prova_ease= EASE_R_Recommender(URM_train= URM_train)
prova_ease.fit(topK= 50, l2_norm= 25.506976565565683)
    #print("RESULT EASE:")
    #result_df, _ = evaluator_validation.evaluateRecommender(prova_ease)
    #print(result_df)

recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
recommender_ItemKNNCF.fit(topK= 5, shrink= 386, normalize= True, feature_weighting = 'TF-IDF')
    #print("RESULT ItemknnCF:")
    #result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF)
    #print(result_df)


from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
RP3_rec= RP3betaRecommender(URM_train)
RP3_rec.fit(alpha= 0.4047398296217158, beta= 0.24691965877972694, min_rating= 0, topK= 12, implicit=True, normalize_similarity= True)
    #print("RESULT new_rp3  not stacked:")
    #result_df, _ = evaluator_validation.evaluateRecommender(RP3_rec)
    #print(result_df)

from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
#0.05173739618478235 
#  parameters: {'ICM_weight': 0.18695141338288582, 'topK': 6, 
# 'shrink': 0, 'normalize': True, 'feature_weighting': 'BM25'}
recommender_itemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train,ICM_matrix)
recommender_itemKNNCFCBF.fit(ICM_weight= 0.18695141338288582, topK= 6, shrink= 0, normalize= True, feature_weighting= 'BM25')

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
P3alpha = P3alphaRecommender(URM_train)
P3alpha.fit(topK= 20, alpha= 0.5559844162882982, normalize_similarity= True)


alpha_par=0.5121359858878799
beta_par= 0.11892301835593802
gamma_par= 0.30116003961682886
recommender_object=ItemKNNCustomSimilarityRecommender(URM_train)

new_similarity = (1 - alpha_par- beta_par - gamma_par) * recommender_itemKNNCFCBF.W_sparse + alpha_par *  RP3_rec.W_sparse + beta_par* recommender_ItemKNNCF.W_sparse + gamma_par* P3alpha.W_sparse

recommender_object.fit(new_similarity)
    


Last_model = DifferentLossScoresHybridRecommender(URM_train,recommender_object, prova_ease)
Last_model.fit(alpha= 0.4347252990297249, norm= 1)



optuna_study.optimize(objective_function,
                      callbacks=[save_results],
                      n_trials = 500,
                      n_jobs=2)

print(optuna_study.best_trial.params)






