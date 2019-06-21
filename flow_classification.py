from logger import Logger
from utils import load_data

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
import pandas as pd
import numpy as np
import gc

from bonus import create_adversarial_examples


def get_train_test_data(df, logger):

  df_selected = df[['ip_src', 'duration', 'protocol', 'flags', 'packets', 'bytes', 'label']]
  logger.log("Select columns for classification: {}".format(df_selected.columns))

  df_selected = df_selected[df_selected['label'] != "Background"]
  logger.log("Remove background flows")

  df_selected['protocol'] = pd.Categorical(df_selected['protocol']).codes
  df_selected['flags'] = pd.Categorical(df_selected['flags']).codes
  df_selected.replace({'label': {"LEGITIMATE": 0, "Botnet": 1}}, inplace = True)
  logger.log("Convert string columns to numerical codes")

  infected_host_train = logger.config_dict['INFECTED_HOSTS'][:5]
  infected_host_test  = logger.config_dict['INFECTED_HOSTS'][5:]
  df_infected_host_train = df_selected[df_selected['ip_src'].isin(infected_host_train)]
  df_infected_host_test  = df_selected[df_selected['ip_src'].isin(infected_host_test)]
  logger.log("Split infected host: first 5 in train, last 5 in test")

  df_normal_host = df_selected[~df_selected['ip_src'].isin(infected_host_train)]
  df_normal_host = df_normal_host[~df_normal_host['ip_src'].isin(infected_host_test)]
  logger.log("Select normal traffic")

  X_normal = df_normal_host.loc[:, df_normal_host.columns != 'label'].values
  y_normal = df_normal_host['label'].values

  X_infected_host_train = df_infected_host_train.loc[:, df_infected_host_train.columns != 'label'].values
  y_infected_host_train = df_infected_host_train['label'].values

  X_infected_host_test = df_infected_host_test.loc[:, df_infected_host_test.columns != 'label'].values
  y_infected_host_test = df_infected_host_test['label'].values

  X_train, X_test, y_train, y_test = train_test_split(X_normal, y_normal, 
    test_size = 0.21, random_state = 13)

  X_train = np.concatenate([X_train, X_infected_host_train])
  y_train = np.concatenate([y_train, y_infected_host_train])

  indices = np.arange(X_train.shape[0])
  np.random.shuffle(indices)
  X_train = X_train[indices]
  y_train = y_train[indices]

  X_test = np.concatenate([X_test, X_infected_host_test])
  y_test = np.concatenate([y_test, y_infected_host_test])

  logger.log("Shuffle and combine in train {} and test {}".format(
    X_train.shape[0], X_test.shape[0]))
  logger.log("Train label distribution {}".format(Counter(y_train)))

  return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train, is_imbalanced, logger):

  if is_imbalanced:
    sample_weight = [1.3 if int(sample) == 1 else 1 for sample in y_train]
  else:
    sample_weight = [1 for _ in y_train]

  X_train   = [elem[1:] for elem in X_train]

  logger.log("Start training {}...".format(type(model).__name__))
  model.fit(X_train, y_train, sample_weight = sample_weight)
  logger.log("Finished training", show_time = True)

  return model


def test_model(model, X_test, y_test, threshold, logger):

  ips_test = [elem[0] for elem in X_test]
  X_test   = [elem[1:] for elem in X_test]

  y_pred = model.predict_proba(X_test)
  y_pred = y_pred[:, 1]
  y_pred = y_pred > threshold
  
  logger.log("Threshold {}".format(threshold))
  logger.log("Packet level", tabs = 1)
  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  logger.log("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp), tabs = 2)
  logger.log("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)), tabs = 2)
  logger.log("Precision: {:.4f}".format(precision_score(y_test, y_pred)), tabs = 2)
  logger.log("Recall: {:.4f}".format(recall_score(y_test, y_pred)), tabs = 2)

  y_pred_host = []
  unique_ips_test = np.unique(ips_test) 
  for ip in unique_ips_test:
    not_infected_in_any_flow = True
    for i, pred in enumerate(y_pred):
      if pred == 1 and ips_test[i] == ip:
        y_pred_host.append(1)
        not_infected_in_any_flow = False
        break
    if not_infected_in_any_flow:
      y_pred_host.append(0)

  y_true_host = [1 if ip in logger.config_dict['INFECTED_HOSTS'] else 0 for ip in unique_ips_test]
  
  logger.log("Host level", tabs = 1)
  tn, fp, fn, tp = confusion_matrix(y_true_host, y_pred_host).ravel()
  logger.log("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp), tabs = 2)
  logger.log("Accuracy: {:.4f}".format(accuracy_score(y_true_host, y_pred_host)), tabs = 2)
  logger.log("Precision: {:.4f}".format(precision_score(y_true_host, y_pred_host)), tabs = 2)
  logger.log("Recall: {:.4f}".format(recall_score(y_true_host, y_pred_host)), tabs = 2)


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")
  df = load_data("DATA_FILE2", logger)

  X_train, y_train, X_test, y_test = get_train_test_data(df, logger)

  df = df[0:0]
  del df
  gc.collect()

  model = LogisticRegression(solver = 'lbfgs', max_iter = 500)
  model = RandomForestClassifier(n_estimators = 200, min_samples_split = 4, min_samples_leaf = 2, 
    max_depth = 9, criterion = "gini", random_state = 13, n_jobs = -1)

  model = train_model(model, X_train, y_train, True, logger)
  test_model(model, X_test, y_test, 0.5, logger)
  test_model(model, X_test, y_test, 0.7, logger)
  test_model(model, X_test, y_test, 0.9, logger)
  test_model(model, X_test, y_test, 0.95, logger)
  test_model(model, X_test, y_test, 0.99, logger)
  test_model(model, X_test, y_test, 0.999, logger)

  duration_alterations = [+1, 0, 0, +1, +10, +45, +120, +120, 0, 0]
  packets_alterations  = [0, +1, 0, +1, +10, +30, +100, 0, +100, 0]
  bytes_alterations    = [0, 0, +1, +1, +16, +256, +1024, 0, 0, +1024]

  for alteration in zip(duration_alterations, packets_alterations, bytes_alterations):
    X_test_new = create_adversarial_examples(X_test, y_test, alter_packets = alteration[1],
      alter_bytes = alteration[2], alter_duration = alteration[0], logger = logger)
    test_model(model, X_test_new, y_test, 0.95, logger)