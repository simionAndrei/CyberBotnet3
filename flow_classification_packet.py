from logger import Logger
from utils import load_data

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import time

import pandas as pd
import numpy as np


def evaluate_model(model, k_folds, X, y, is_imbalanced, logger):

  logger.log("Start {}-fold cross validation on {} entries".format(
    k_folds, y.shape[0]))
    
  start = time.time()

  strat_kfold = StratifiedKFold(n_splits= k_folds, shuffle=True, random_state = 13)

  acc_list, prec_list, rec_list, confmat_list = [[] for _ in range(4)]
  for fold_idx, (train_idx, test_idx) in enumerate(strat_kfold.split(X, y)):

    crt_X_train, crt_X_test = X[train_idx], X[test_idx]
    crt_y_train, crt_y_test = y[train_idx], y[test_idx]

    if is_imbalanced:
      sample_weight = [2 if int(sample) == 1 else 1 for sample in crt_y_train]
    else:
      sample_weight = [1 for _ in crt_y_train]

    model.fit(crt_X_train, crt_y_train, sample_weight = sample_weight)

    y_pred = model.predict_proba(crt_X_test)
    y_pred = y_pred[:, 1]
    y_pred = y_pred > 0.5

    tn, fp, fn, tp = confusion_matrix(crt_y_test, y_pred).ravel()
    acc_list.append(accuracy_score(crt_y_test, y_pred))
    prec_list.append(precision_score(crt_y_test, y_pred))
    rec_list.append(recall_score(crt_y_test, y_pred))
    confmat_list.append([tn, fp, fn, tp])

    logger.log("TN: {}, FP: {}, FN: {}, TP: {} at fold#{}".format(tn, fp, fn, tp, fold_idx))
    logger.log("Accuracy at fold#{}: {:.5f}".format(fold_idx, acc_list[-1]))
    logger.log("Precision at fold#{}: {:.5f}".format(fold_idx, prec_list[-1]))
    logger.log("Recall at fold#{}: {:.5f}".format(fold_idx, rec_list[-1]))
  
  logger.log("Finished cross validation in {:.2f}s".format(time.time() - start))
  logger.log("Mean accuracy: {:.5f}".format(np.average(acc_list)))
  logger.log("Mean precision: {:.5f}".format(np.average(prec_list)))
  logger.log("Mean recall: {:.5f}".format(np.average(rec_list)))
  logger.log("Mean FP: {}".format(np.average([e[1] for e in confmat_list])))
  logger.log("Mean TP: {}".format(np.average([e[3] for e in confmat_list])))
  logger.log("Mean FN: {}".format(np.average([e[2] for e in confmat_list])))
  logger.log("Mean TN: {}".format(np.average([e[0] for e in confmat_list])))


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  df = load_data("DATA_FILE2", logger)

  df_selected = df[['duration', 'protocol', 'flags', 'packets', 'bytes', 'label']]

  df_selected = df_selected[df_selected['label'] != "Background"]

  df_selected['protocol'] = pd.Categorical(df_selected['protocol']).codes
  df_selected['flags'] = pd.Categorical(df_selected['flags']).codes
  df_selected.replace({'label': {"LEGITIMATE": 0, "Botnet": 1}}, inplace = True)

  X = df_selected.loc[:, df_selected.columns != 'label'].values
  y = df_selected['label'].values

  logger.log("Data distribution {} Legitimate and {} 1 Botnet".format(sum(y == 0), sum(y == 1)))

  model = LogisticRegression(solver = 'lbfgs', max_iter = 500)
  model = RandomForestClassifier(n_estimators = 200, n_jobs = -1)
  evaluate_model(model, 10, X, y, False, logger)