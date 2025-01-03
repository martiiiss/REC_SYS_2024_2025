import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import matplotlib.pyplot as plt
from numpy import linalg as LA
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.Recommender_import_list import *
import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from sklearn.metrics import average_precision_score
from tqdm import tqdm
# Carica il file CSV  -> URM
data = pd.read_csv('data_train.csv')

# Trova il massimo user_id e item_id per dimensionare correttamente la matrice
max_user_id = data['user_id'].max()
max_item_id = data['item_id'].max()

# Crea la matrice sparsa
URM_all = csr_matrix((data['data'], (data['user_id'], data['item_id'])), shape=(max_user_id + 1, max_item_id + 1))
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage = 0.8)
# Calcola il numero di utenti unici
num_users = URM_all.shape[0]


# Assuming URM_train is your User-Rating Matrix
profile_length = np.ediff1d(sps.csr_matrix(URM_all).indptr)  # Number of interactions for each user

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



# Verifica se ICM_matrix ha lo stesso numero di righe di URM_train, altrimenti riempie le righe mancanti
if ICM_matrix.shape[0] < URM_train.shape[1]:
    ICM_matrix = sps.vstack([ICM_matrix, sps.csr_matrix((URM_train.shape[1] - ICM_matrix.shape[0], ICM_matrix.shape[1]))])

# Concatenate URM and ICM
stacked_URM = sps.vstack([URM_train, ICM_matrix.T])
stacked_URM = sps.csr_matrix(stacked_URM)

# Print the concatenated stacked_URM matrix
print("\nstacked_URM (after concatenation):")
#print(stacked_URM)

from Evaluation.Evaluator import EvaluatorHoldout
cutoff_list=[10]

evaluator_validation = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

from Recommenders.BaseRecommender import BaseRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Evaluation.Evaluator import EvaluatorHoldout

"""
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
prova_ease= EASE_R_Recommender(URM_train= URM_train)
prova_ease.fit(topK= 95, l2_norm= 20.982665537584804)
print("RESULT EASE:")
result_df, _ = evaluator_validation.evaluateRecommender(prova_ease)
print(result_df)
"""

recommender_ItemKNNCF = ItemKNNCFRecommender(URM_train)
recommender_ItemKNNCF.fit(topK= 5, shrink= 386, normalize= True, feature_weighting = 'TF-IDF')
#print("RESULT ItemknnCF:")
#result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCF)
#print(result_df)

"""
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
RP3_rec= RP3betaRecommender(URM_train)
RP3_rec.fit(alpha= 0.4047398296217158, beta= 0.24691965877972694, min_rating= 0, topK= 12, implicit=True, normalize_similarity= True)
print("RESULT new_rp3  not stacked:")
result_df, _ = evaluator_validation.evaluateRecommender(RP3_rec)
print(result_df)


from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
#0.05173739618478235 
#  parameters: {'ICM_weight': 0.18695141338288582, 'topK': 6, 
# 'shrink': 0, 'normalize': True, 'feature_weighting': 'BM25'}
recommender_itemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train,ICM_matrix)
recommender_itemKNNCFCBF.fit(ICM_weight= 0.18695141338288582, topK= 6, shrink= 0, normalize= True, feature_weighting= 'BM25')

alpha_par=0.9408644746147785
beta_par= 0.20036203249355136
recommender_object=ItemKNNCustomSimilarityRecommender(URM_train)
new_similarity = (1 - alpha_par) * recommender_itemKNNCFCBF.W_sparse + alpha_par *  RP3_rec.W_sparse + beta_par* recommender_ItemKNNCF.W_sparse
recommender_object.fit(new_similarity)
"""

from Recommenders.BaseRecommender import BaseRecommender

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
        
        
        
    def fit(self, norm, alpha = 0.5):

        self.alpha = alpha
        self.norm = norm


    def _compute_item_score(self, user_id_array, items_to_compute):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights


#Last_model = DifferentLossScoresHybridRecommender(URM_train,recommender_object, prova_ease)
#Last_model.fit(alpha=0.27985586193900014, norm= -np.inf)


#Slim_Elasticnet= SLIMElasticNetRecommender(URM_train)
#Slim_Elasticnet.fit(l1_ratio= 0.174358035009766, topK= 631, alpha= 0.00010361443114176747)


from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
candidate_generator_recommender = recommender_ItemKNNCF
#DifferentLossScoresHybridRecommender(URM_train,recommender_ItemKNNCF,Last_model,Slim_Elasticnet)
#candidate_generator_recommender.fit(alpha= 0.2796027263555756, norm= 2)
#print("RESULT TOTAL:")
#result_df, _ = evaluator_validation.evaluateRecommender(candidate_generator_recommender)
#print(result_df)



import pandas as pd
from tqdm import tqdm
import scipy.sparse as sps
import numpy as np
from xgboost import XGBRanker

n_users, n_items = URM_train.shape

training_dataframe = pd.DataFrame(index=range(0,n_users), columns = ["ItemID"])
training_dataframe.index.name='UserID'

cutoff = 30

for user_id in tqdm(range(n_users)):    
    recommendations = candidate_generator_recommender.recommend(user_id, cutoff = cutoff)
    training_dataframe.loc[user_id, "ItemID"] = recommendations

training_dataframe = training_dataframe.explode("ItemID")

URM_validation_coo = sps.coo_matrix(URM_validation)

correct_recommendations = pd.DataFrame({"UserID": URM_validation_coo.row,
                                        "ItemID": URM_validation_coo.col})
print(correct_recommendations)

training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID','ItemID'], how='left', indicator='Exist')

training_dataframe["Label"] = training_dataframe["Exist"] == "both"
training_dataframe.drop(columns = ['Exist'], inplace=True)
print(training_dataframe)


from Recommenders.NonPersonalizedRecommender import TopPop
topp= TopPop(URM_train)
topp.fit()

itemKNN_CF_CBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train,ICM_matrix)
itemKNN_CF_CBF.fit(ICM_weight= 0.18695141338288582, topK= 6, shrink= 0, normalize= True, feature_weighting= 'BM25')

from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
p3alpha= P3alphaRecommender(URM_train)
p3alpha.fit(topK= 20, alpha= 0.5559844162882982, normalize_similarity= True)

other_algorithms = {
    "TopPop": topp,
    "P3Alpha":p3alpha,
    "itemKNN_CF_CBF": itemKNN_CF_CBF,

}

training_dataframe = training_dataframe.set_index('UserID')

for user_id in tqdm(range(n_users)):  
    for rec_label, rec_instance in other_algorithms.items():
        
        item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
        
        all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute = item_list)

        training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list] 

training_dataframe = training_dataframe.reset_index()
training_dataframe = training_dataframe.rename(columns = {"index": "UserID"})

item_popularity = np.ediff1d(sps.csc_matrix(URM_train).indptr)
training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)] 

user_popularity = np.ediff1d(sps.csr_matrix(URM_train).indptr)
training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

print(training_dataframe)

# Ground truth: estrai gli item rilevanti per ogni utente dalla matrice URM_validation
true_recommendations = {}
for user_id in range(URM_validation.shape[0]):
    true_recommendations[user_id] = URM_validation[user_id].indices

# Funzione per calcolare il MAP@10
def compute_map_at_k(y_true, y_pred, k=10):
    map_at_k = []
    for true_items, predicted_items in zip(y_true, y_pred):
        # Trova i top-k predetti
        top_k_pred = predicted_items[:k]
        
        # Calcola il numero di elementi corretti in top-k
        num_hits = 0
        precision_sum = 0.0
        for idx, item in enumerate(top_k_pred):
            if item in true_items:
                num_hits += 1
                precision_sum += num_hits / (idx + 1)
        
        # Calcola la precisione media
        if len(true_items) > 0:
            map_at_k.append(precision_sum / min(len(true_items), k))
        else:
            map_at_k.append(0.0)
    
    return np.mean(map_at_k)


def objective(trial):
    # Hyperparameters to optimize
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-5, 1e-1,log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-5, 1e-1,log=True)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    max_leaves = trial.suggest_int('max_leaves', 0, 50)
    grow_policy = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    objective = trial.suggest_categorical('objective', ['rank:pairwise', 'rank:ndcg'])
    booster = trial.suggest_categorical('booster', ['gbtree', 'dart'])
    
    # Create the XGBRanker model with optimized hyperparameters
    model = XGBRanker(
        objective=objective,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        max_depth=max_depth,
        max_leaves=max_leaves,
        grow_policy=grow_policy,
        booster=booster,
        verbosity=0,
        random_state=42,
        enable_categorical = True
    )
    
    
    # Preprocess training data
    X_train = training_dataframe.drop(columns=["Label"])  # Features
    # Ensure ItemID is integer type (or categorical)
    X_train['ItemID'] = X_train['ItemID'].astype(int)  # Ensure it’s int
    y_train = training_dataframe["Label"]  # Labels
    groups = training_dataframe.groupby("UserID").size().values  # Grouping by user
    # Grouping by user
    
    # Fit the model
    model.fit(X_train, y_train, group=groups)
    
    # Generate recommendations for the test set
    y_pred = []
    y_true = []
    for user_id in tqdm(range(n_users)):
        # Predictions from the recommender
        X = training_dataframe
        X['ItemID']=X['ItemID'].astype(int) 
        X = training_dataframe.drop(columns=["Label"])
        predictions = model.predict(X)
        reranked_dataframe = training_dataframe.copy()
        reranked_dataframe['rating_xgb'] = pd.Series(predictions, index=reranked_dataframe.index)
        reranked_dataframe = reranked_dataframe.sort_values(['UserID','rating_xgb'], ascending=[True, False])
        recommendations_per_user = reranked_dataframe.loc[reranked_dataframe['UserID'] == user_id].ItemID.values[:10]
        y_pred.append(recommendations_per_user)
        y_true.append(set(true_recommendations[user_id]))
    
    # Calculate MAP@10
    map_at_10 = compute_map_at_k(y_true, y_pred, k=10)
    
    return map_at_10

# Impostazione dello studio Optuna
study = optuna.create_study(direction='maximize')  # Ottimizzare MAP@10 (massimizzazione)
study.optimize(objective, n_trials=50)  # Numero di esperimenti di ottimizzazione

# Stampa dei migliori risultati
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Params: {trial.params}")





"""
trace_users = pd.read_csv('data_target_users_test.csv')
trace_users = trace_users.to_numpy()
# Open the file for writing recommendations
with open('sample_submission.csv', 'w') as file:
    
    file.write("user_id,item_list\n")
    # Loop over each user
    for user_id in trace_users:
        # Get the top 10 recommendations for the user
        recommended_items =candidate_generator_recommender.recommend(user_id_array=user_id, cutoff=10)
        stringa=  (' '.join(map(str, np.array(recommended_items))))
        file.write(f"{user_id[0]}, {stringa[1: len(stringa) -1]}\n")
        
print("Recommendations have been written to 'sample_submission.csv'")
"""
