import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import matplotlib.pyplot as plt
from numpy import linalg as LA
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender

# Carica il file CSV  -> URM
data = pd.read_csv('data_train.csv')

# Trova il massimo user_id e item_id per dimensionare correttamente la matrice
max_user_id = data['user_id'].max()
max_item_id = data['item_id'].max()

# Crea la matrice sparsa
URM_all = csr_matrix((data['data'], (data['user_id'], data['item_id'])), shape=(max_user_id + 1, max_item_id + 1))
URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.8)
URM_train= URM_all
# Calcola il numero di utenti unici
num_users = URM_all.shape[0]


# Assuming URM_train is your User-Rating Matrix
profile_length = np.ediff1d(sps.csr_matrix(URM_all).indptr)  # Number of interactions for each user

# Define interaction ranges for each group
group_0_mask = (profile_length == 0) 
group_1_mask = (profile_length >= 1) & (profile_length <= 20) 
group_2_mask = (profile_length >= 20) & (profile_length <= 51)  
group_3_mask = (profile_length > 51)  

# Get users for each group
group_0_users = np.where(group_0_mask)[0] #users with 0 iterations
group_1_users = np.where(group_1_mask)[0]
group_2_users = np.where(group_2_mask)[0]
group_3_users = np.where(group_3_mask)[0]

# Print out the number of users in each group
print(f"Group no iterations: {len(group_0_users)} users")
print(f"Group 1 (1-20 interactions): {len(group_1_users)} users -> best models: SLIMBPR RP3beta P3alpha ItemKNNCFCBF userKNNCF")
print(f"Group 2 (20-51 interactions): {len(group_2_users)} users -> best models: RP3Beta ItemKNNCFCBF SLIMBPR UserKNNCF P3alpha")
print(f"Group 3 (51-... interactions): {len(group_3_users)} users -> best models RP3Beta  ItemKNNCFCBF  UserKNNCF  pureSVD ")

# Switch-like structure for actions based on user group
def get_best_models_for_user(user_id):
    if user_id in group_0_users:
        return 0
    elif user_id in group_1_users:
        return 1
    elif user_id in group_2_users:
        return 2
    elif user_id in group_3_users:
        return 3
    else:
        return 2


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
#print("\nICM_matrix (before concatenation):")
#print(ICM_matrix)

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
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

Slim_Elasticnet= SLIMElasticNetRecommender(URM_train)
Slim_Elasticnet.fit(l1_ratio= 0.174358035009766, topK= 631, alpha= 0.00010361443114176747)
#l1_ratio= 0.7508019942429333, alpha= 4.462842551622246e-05, topK= 197)
#l1_ratio= 0.011453254111306543, topK= 422, alpha= 0.0011524047751432808)
#l1_ratio=0.7042629316804924, alpha=4.364330124793646e-05, topK=237)


from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
prova_ease= EASE_R_Recommender(URM_train= URM_train)
prova_ease.fit(topK= 95, l2_norm= 20.982665537584804)
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

alpha_par=0.9408644746147785
beta_par= 0.20036203249355136
recommender_object=ItemKNNCustomSimilarityRecommender(URM_train)
new_similarity = (1 - alpha_par) * recommender_itemKNNCFCBF.W_sparse + alpha_par *  RP3_rec.W_sparse + beta_par* recommender_ItemKNNCF.W_sparse
recommender_object.fit(new_similarity)



#DO NOT USE IT
#RESULT = 0.022

from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
#{'ICM_weight': 0.5484923216059217, 'topK': 77, 'shrink': 815, 'normalize': True, 'feature_weighting': 'BM25'
#0.028106832262946156
"""
recommender_ItemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(stacked_URM, ICM_matrix)
recommender_ItemKNNCFCBF.fit(ICM_weight= 0.28843762927009053, topK= 117, shrink= 66, normalize =True, feature_weighting= 'TF-IDF')
print("RESULT recommender_ItemKNNCFCBF:")
result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCFCBF)
print(result_df)
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


 

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        
        item_weights_1 = self.recommender_1._compute_item_score(user_id_array,items_to_compute=items_to_compute)
        item_weights_2 = self.recommender_2._compute_item_score(user_id_array,items_to_compute=items_to_compute)

        norm_item_weights_1 = LA.norm(item_weights_1, self.norm)
        norm_item_weights_2 = LA.norm(item_weights_2, self.norm)
        
        
        if norm_item_weights_1 == 0:
            raise ValueError("Norm {} of item weights for recommender 1 is zero. Avoiding division by zero".format(self.norm))
        
        if norm_item_weights_2 == 0:
            raise ValueError("Norm {} of item weights for recommender 2 is zero. Avoiding division by zero".format(self.norm))
        
        item_weights = item_weights_1 / norm_item_weights_1 * self.alpha + item_weights_2 / norm_item_weights_2 * (1-self.alpha)

        return item_weights






from Recommenders.KNN.ItemKNN_CFCBF_Hybrid_Recommender import ItemKNN_CFCBF_Hybrid_Recommender
"""
recommender_ItemKNNCFCBF = ItemKNN_CFCBF_Hybrid_Recommender(URM_train, ICM_matrix)
recommender_ItemKNNCFCBF.fit(ICM_weight= 0.13196763624825714, topK= 6, shrink= 765, normalize= True, feature_weighting= 'TF-IDF')
print("RESULT knncfcbf:")
result_df, _ = evaluator_validation.evaluateRecommender(recommender_ItemKNNCFCBF)
print(result_df)
"""
#RESULT =0.045 if rp3 not stacked
#RESULT = 0.055791 with alpha =0.4 on(P3alpha, RP3_rec)
#RESULT = 0.055969 WITH (,P3alpha, RP3_rec, recommender_object)
# {'alpha': 0.001968760016611275, 'beta': 0.5662926367392187, 'gamma': 0.972902764598479}
#RESULT= 0.05549  WITH (URM_train,P3alpha, RP3_rec)
#alpha= 1.7197811097773745, beta= 1.8078553339774344)


Last_model = DifferentLossScoresHybridRecommender(URM_train,recommender_object, prova_ease)
Last_model.fit(alpha=0.27985586193900014, norm= -np.inf)



scoreshybridrecommender= DifferentLossScoresHybridRecommender(URM_train,Last_model,Slim_Elasticnet)
scoreshybridrecommender.fit(alpha= 0.2796027263555756, norm= 2)
#TOO  MANY HYPER PARAMETERS TO ADJUST
"""
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
recommender_IALS = IALSRecommender(URM_train)
recommender_IALS.fit(num_factors= 154, epochs= 140, confidence_scaling= 'linear', alpha=  6.28182586673945, epsilon= 10.0, reg= 1e-05)
print("RESULT IALS:")
result_df, _ = evaluator_validation.evaluateRecommender(recommender_IALS)
print(result_df)
"""





trace_users = pd.read_csv('data_target_users_test.csv')
trace_users = trace_users.to_numpy()
# Open the file for writing recommendations
with open('sample_submission.csv', 'w') as file:
    
    file.write("user_id,item_list\n")
    # Loop over each user
    for user_id in trace_users:
        # Get the top 10 recommendations for the user
        recommended_items = scoreshybridrecommender.recommend(user_id_array=user_id, cutoff=10)
        stringa=  (' '.join(map(str, np.array(recommended_items))))
        file.write(f"{user_id[0]}, {stringa[1: len(stringa) -1]}\n")
        
print("Recommendations have been written to 'sample_submission.csv'")

