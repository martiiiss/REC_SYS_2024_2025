#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22/11/17

@author: Maurizio Ferrari Dacrema
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sps
import matplotlib.pyplot as plt
from Recommenders.Recommender_import_list import *

import traceback

import os, multiprocessing
from functools import partial

from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Data_manager.split_functions.split_train_validation_random_holdout import split_train_in_two_percentage_global_sample

from HyperparameterTuning.run_hyperparameter_search import runHyperparameterSearch_Collaborative, runHyperparameterSearch_Content, runHyperparameterSearch_Hybrid


def read_data_split_and_search():
    """
    This function provides a simple example on how to tune parameters of a given algorithm

    The BayesianSearch object will save:
        - A .txt file with all the cases explored and the recommendation quality
        - A _best_model file which contains the trained model and can be loaded with recommender.load_model()
        - A _best_parameter file which contains a dictionary with all the fit parameters, it can be passed to recommender.fit(**_best_parameter)
        - A _best_result_validation file which contains a dictionary with the results of the best solution on the validation
        - A _best_result_test file which contains a dictionary with the results, on the test set, of the best solution chosen using the validation set
    """

        # Carica il file CSV  -> URM
    data = pd.read_csv('data_train.csv')

    # Trova il massimo user_id e item_id per dimensionare correttamente la matrice
    max_user_id = data['user_id'].max()
    max_item_id = data['item_id'].max()

    # Crea la matrice sparsa
    URM_all = csr_matrix((data['data'], (data['user_id'], data['item_id'])), shape=(max_user_id + 1, max_item_id + 1))
    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage = 0.8)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage = 0.8)

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


    output_folder_path = "result_experiments/"


    # If directory does not exist, create
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)


    collaborative_algorithm_list = [
        EASE_R_Recommender
    ]




    from Evaluation.Evaluator import EvaluatorHoldout

    cutoff_list = [5, 10, 20]
    metric_to_optimize = "MAP"
    cutoff_to_optimize = 10

    n_cases = 100
    n_random_starts = int(n_cases/3)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list = cutoff_list)
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list = cutoff_list)


    runParameterSearch_Collaborative_partial = partial(runHyperparameterSearch_Collaborative,
                                                       URM_train = URM_train,
                                                       metric_to_optimize = metric_to_optimize,
                                                       cutoff_to_optimize = cutoff_to_optimize,
                                                       n_cases = n_cases,
                                                       n_random_starts = n_random_starts,
                                                       evaluator_validation_earlystopping = evaluator_validation,
                                                       evaluator_validation = evaluator_validation,
                                                       evaluator_test = evaluator_test,
                                                       output_folder_path = output_folder_path,
                                                       resume_from_saved = True,
                                                       similarity_type_list = ["cosine"],
                                                       parallelizeKNN = False)





    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count()), maxtasksperchild=1)
    pool.map(runParameterSearch_Collaborative_partial, collaborative_algorithm_list)

    #
    #
    # for recommender_class in collaborative_algorithm_list:
    #
    #     try:
    #
    #         runParameterSearch_Collaborative_partial(recommender_class)
    #
    #     except Exception as e:
    #
    #         print("On recommender {} Exception {}".format(recommender_class, str(e)))
    #         traceback.print_exc()
    #




    ################################################################################################
    ###### Content Baselines

    for ICM_name, ICM_object in dataset.get_loaded_ICM_dict().items():

        try:

            runHyperparameterSearch_Content(ItemKNNCBFRecommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)

        except Exception as e:

            print("On CBF recommender for ICM {} Exception {}".format(ICM_name, str(e)))
            traceback.print_exc()


        try:

            runHyperparameterSearch_Hybrid(ItemKNN_CFCBF_Hybrid_Recommender,
                                        URM_train = URM_train,
                                        URM_train_last_test = URM_train + URM_validation,
                                        metric_to_optimize = metric_to_optimize,
                                        cutoff_to_optimize = cutoff_to_optimize,
                                        evaluator_validation = evaluator_validation,
                                        evaluator_test = evaluator_test,
                                        output_folder_path = output_folder_path,
                                        parallelizeKNN = True,
                                        allow_weighting = True,
                                        resume_from_saved = True,
                                        similarity_type_list = ["cosine"],
                                        ICM_name = ICM_name,
                                        ICM_object = ICM_object.copy(),
                                        n_cases = n_cases,
                                        n_random_starts = n_random_starts)


        except Exception as e:

            print("On recommender {} Exception {}".format(ItemKNN_CFCBF_Hybrid_Recommender, str(e)))
            traceback.print_exc()





if __name__ == '__main__':


    read_data_split_and_search()
