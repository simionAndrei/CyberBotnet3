import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def get_train_test_data(df, logger):
    df_selected = df[['ip_src', 'duration', 'protocol', 'flags', 'packets', 'bytes', 'label']]
    logger.log("Select columns for classification: {}".format(df_selected.columns))

    df_selected = df_selected[df_selected['label'] != "Background"]
    logger.log("Remove background flows")

    df_selected['protocol'] = pd.Categorical(df_selected['protocol']).codes
    df_selected['flags'] = pd.Categorical(df_selected['flags']).codes
    df_selected.replace({'label': {"LEGITIMATE": 0, "Botnet": 1}}, inplace=True)
    logger.log("Convert string columns to numerical codes")

    infected_host_train = logger.config_dict['INFECTED_HOSTS'][:5]
    infected_host_test = logger.config_dict['INFECTED_HOSTS'][5:]
    df_infected_host_train = df_selected[df_selected['ip_src'].isin(infected_host_train)]
    df_infected_host_test = df_selected[df_selected['ip_src'].isin(infected_host_test)]
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
                                                        test_size=0.21, random_state=13)

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


def extract_other(data_df, self_ip):
    series = [data_df[data_df['ip_src'] == self_ip]['ip_dest'],
              data_df[data_df['ip_dest'] == self_ip]['ip_src']]

    ips = pd.concat(series)

    return ips


def true_count(ips):
    ip_occurences = Counter(ips.values).items()
    ip_occurences = sorted(ip_occurences, key=lambda x: (x[1], x[0]), reverse=True)

    return ip_occurences


def get_estimation_error(true_freq, estimated_freq, estimation_size):
    freq_err = 0
    for ip, freq in true_freq[:estimation_size]:
        if ip in list(zip(*estimated_freq[:estimation_size]))[0]:
            freq_err += abs(freq - dict(estimated_freq[:estimation_size])[ip])
        else:
            freq_err += freq

    return freq_err


def print_stats(true, predicted, logger):
    tn, fp, fn, tp = confusion_matrix(true, predicted).ravel()
    logger.log("TN: {}, FP: {}, FN: {}, TP: {}".format(tn, fp, fn, tp), tabs=2)
    logger.log("Accuracy: {:.4f}".format(accuracy_score(true, predicted)), tabs=2)
    logger.log("Precision: {:.4f}".format(precision_score(true, predicted)), tabs=2)
    logger.log("Recall: {:.4f}".format(recall_score(true, predicted)), tabs=2)


def plot_estimation_error(x, values, xlabel, plt_title, filename, logger):
    sns.set()
    plt.figure(figsize=(6, 6))

    plt.title(plt_title)
    plt.xlabel(xlabel)
    plt.ylabel("Estimation error")
    plt.plot(x, values, marker="X")

    plt.savefig(logger.get_output_file(filename), dpi=120,
                bbox_inches='tight')


def load_data(data_key, logger):
    colnames = ["date", "duration", "protocol", "ip_src", "port_src", "ip_dest",
                "port_dest", "flags", "tos", "packets", "bytes", "flows", "label"]

    data_file = logger.config_dict[data_key]
    logger.log("Start loading data from {}...".format(data_file))

    file_content = []
    with open(logger.get_data_file(data_file), "r") as fp:
        for line_idx, line_content in enumerate(fp.readlines()):
            crt_content = []
            if line_idx == 0:
                continue

            line_list = line_content.strip().split()
            crt_content.append(line_list[0] + " " + line_list[1])  # date
            crt_content.append(line_list[2])  # duration
            crt_content.append(line_list[3])  # protocol
            crt_content.append(line_list[4].split(':')[0])  # ip_src
            crt_content.append(None if len(line_list[4].split(':')) == 1 else line_list[4].split(':')[1])  # port_src
            crt_content.append(line_list[6].split(':')[0])  # ip_dest
            crt_content.append(None if len(line_list[6].split(':')) == 1 else line_list[6].split(':')[1])  # port_dest
            crt_content.append(line_list[7])  # flags
            crt_content.append(line_list[8])  # tos
            crt_content.append(line_list[9])  # packets
            crt_content.append(line_list[10])  # bytes
            crt_content.append(line_list[11])  # flows
            crt_content.append(line_list[12])  # label

            file_content.append(crt_content)

            if line_idx % 100000 == 0:
                logger.log("Line {}: {}".format(line_idx, crt_content))

    df = pd.DataFrame(file_content, columns=colnames)
    df['date'] = pd.to_datetime(df['date'])
    df['packets'] = pd.to_numeric(df['packets'])
    df['bytes'] = pd.to_numeric(df['bytes'])
    df['duration'] = pd.to_numeric(df['duration'])
    df.sort_values(by=['date'], inplace=True)

    logger.log("Finished loading file", show_time=True)

    return df


if __name__ == "__main__":
    print("Library module. Not main function.")
