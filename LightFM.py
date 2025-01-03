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

# to avoid deprecate problem
np.float = float
np.int = int


class ScoresHybridRecommender(BaseRecommender):
   

    RECOMMENDER_NAME = "ScoresHybridRecommender"

    def __init__(self, URM_train, recommender_1, recommender_2):
        super(ScoresHybridRecommender, self).__init__(URM_train)

        self.URM_train = sps.csr_matrix(URM_train)
        self.recommender_1 = recommender_1
        self.recommender_2 = recommender_2
       
    def fit(self, alpha = 0.5):
        self.alpha = alpha      
        
    def _compute_item_score(self, user_id_array, items_to_compute):
        
        # In a simple extension this could be a loop over a list of pretrained recommender objects
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)
        item_weights = item_weights_1*self.alpha + item_weights_2* (1 -self.alpha)

        return item_weights



###################### LOAD DATA ######################
# Load URM data
data = pd.read_csv('data_train.csv')

# Find max user_id and item_id
max_user_id = data['user_id'].max()
max_item_id = data['item_id'].max()

# Create a sparse URM matrix
URM_all = csr_matrix((data['data'].astype(bool),
                      (data['user_id'], data['item_id'])),
                     shape=(max_user_id + 1, max_item_id + 1))

# Split dataset
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)

# Load ICM data
data_ICM = pd.read_csv('data_ICM_metadata.csv')

# Evaluator setup
evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=[10])


# Extract required columns
item_ids = data_ICM['item_id'].values
feature_ids = data_ICM['feature_id'].values
data_values = data_ICM['data'].values

# Determine matrix size
num_items = item_ids.max() + 1
num_features = feature_ids.max() + 1

# Create the sparse ICM matrix
ICM_matrix = sps.csr_matrix((data_values.astype(bool),
                             (item_ids, feature_ids)),
                             shape=(num_items, num_features))

# Ensure ICM and URM compatibility
if ICM_matrix.shape[0] < URM_train.shape[1]:
    padding = URM_train.shape[1] - ICM_matrix.shape[0]
    ICM_matrix = sps.vstack([ICM_matrix, sps.csr_matrix((padding, ICM_matrix.shape[1]), dtype=bool8)])

# Stack URM and ICM matrices
stacked_URM = sps.vstack([URM_train, ICM_matrix.T], format='csr', dtype=bool)


###################### HYPERPARAMETER OPTIMIZATION ######################
# Optuna optimization
class SaveResults:
    def _init_(self, save_path="results.csv"):
        self.save_path = save_path
        self.results_df = pd.DataFrame(columns=["result"])

    def _call_(self, optuna_study, optuna_trial):
        hyperparam_dict = optuna_trial.params.copy()
        hyperparam_dict["result"] = optuna_trial.values[0]

        self.results_df = self.results_df._append(hyperparam_dict, ignore_index=True)
        self.results_df.to_csv(self.save_path, index=False)

def objective_function(optuna_trial):
    '''LightFM'''
    # Define the hyperparameters to optimize
    n_components = optuna_trial.suggest_int("n_components", 1, 200)
    learning_rate = optuna_trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True)
    item_alpha = optuna_trial.suggest_float("item_alpha", 1e-5, 1e-2, log=True)
    user_alpha = optuna_trial.suggest_float("user_alpha", 1e-5, 1e-2, log=True)
    loss = optuna_trial.suggest_categorical("loss", ['bpr', 'warp', 'warp-kos'])
    epochs = optuna_trial.suggest_int("epochs",300,300)  # Add epochs as a hyperparameter

    # Initialize the LightFM model
    model = LightFMItemHybridRecommender(
        URM_train= URM_train,
        ICM_train=item_features,
    )

    # Train the model
    model.fit(
        n_components=n_components,
        loss=loss,
        learning_rate=learning_rate,
        item_alpha=item_alpha,
        user_alpha= user_alpha,
        epochs=epochs,  # Use the suggested number of epochs
    )

    # Evaluate the recommender using the evaluator_validation
    result_df, _ = evaluator_validation.evaluateRecommender(model)

    # Return MAP for cutoff=10 (or whatever cutoff is needed)
    map_value = result_df.loc[10]["MAP"]
    print(f"Trial MAP: {map_value:.4f}")

    # Forza la garbage collection
    del recommender_instance
    gc.collect()

    return map_value


################## BASE model ##################
#-----------MODEL 1
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
recommender_ItemKNNCF.fit(topK= 5, shrink= 386, normalize= True, feature_weighting = 'TF-IDF')
#print("RESULT ItemknnCF:")
#result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF)
#print(result_df)

#-----------MODEL 2
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
RP3_rec= RP3betaRecommender(URM_train)
RP3_rec.fit(alpha= 0.4047398296217158, beta= 0.24691965877972694, min_rating= 0, topK= 12, implicit=True, normalize_similarity= True)
#print("RESULT new_rp3  not stacked:")
#result_df, _ = evaluator_validation.evaluateRecommender(RP3_rec)
#print(result_df)

#----------MODEL 3
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
recommender_object = ItemKNNCustomSimilarityRecommender(URM_train)
alpha_par= 0.78
#print("alpha=", alpha_par)   
new_similarity = (1 - alpha_par) * recommender_ItemKNNCF.W_sparse + alpha_par *  RP3_rec.W_sparse 
recommender_object.fit(new_similarity)
#print("RESULT KNN CUSTOM SIMILARITY")
#result_df, _ = evaluator_validation.evaluateRecommender(recommender_object)
#print(result_df)

#---------MODEL 4

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
prova_ease= EASE_R_Recommender(URM_train= URM_train)
#prova_ease.fit(topK= 95, l2_norm= 20.982665537584804)
#print("RESULT EASE:")
#result_df, _ = evaluator_validation.evaluateRecommender(prova_ease)
#print(result_df)


recommender_instance = ScoresHybridRecommender(URM_train,recommender_object, recommender_object)
alpha_par= 0.51
#print("alpha=", alpha_par)
recommender_instance.fit(alpha= alpha_par)
#print("RESULT TOTAL:")
#result_df, _ = evaluator_validation.evaluateRecommender(scoreshybridrecommender)
#print(result_df)

# Base_scores: calcola le raccomandazioni per tutti gli utenti e per tutti gli item
Base_scores = np.vstack([
    recommender_instance._compute_item_score(np.array([user_id]), items_to_compute=np.arange(ICM_matrix.shape[0]))
    for user_id in range(URM_train.shape[0])
])

# Verifica della forma originale
print(f"Original shape of Base_scores: {Base_scores.shape}")  # Expected (num_users, num_items)

# Rimuove dimensioni extra, se presenti
Base_scores = np.squeeze(Base_scores)

# Converti Base_scores in matrice sparsa
item_features = csr_matrix(Base_scores.T)  # Transpose to (num_items, num_users)

# Verifica la forma della matrice sparsa
print(f"Item features shape (sparse): {item_features.shape}")  # Expected (num_items, num_users)


#Prepare Interaction Data
interactions = csr_matrix(URM_train)


################## lIGHTFM model ##################
from Recommenders.FactorizationMachines.LightFMRecommender import LightFMItemHybridRecommender
#from lightfm.datasets import fetch_movielens


optuna_study = optuna.create_study(direction="maximize", study_name="lIGHTFM_OPT",
                                   storage="sqlite:///optuna_study.db", load_if_exists=True)

save_results = SaveResults()

optuna_study.optimize(objective_function, callbacks=[save_results], n_trials=50)

print("Best parameters:", optuna_study.best_trial.params)