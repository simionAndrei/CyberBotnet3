import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def extract_other(data_df, self_ip):
    series = [data_df[data_df['ip_src'] == self_ip]['ip_dest'],
              data_df[data_df['ip_dest'] == self_ip]['ip_src']]

    ips = pd.concat(series)

    return ips


def true_count(ips):
    ip_occurences = Counter(ips.values).items()

    d = {}

    for ip, occurences in ip_occurences:
        d[ip] = occurences

    return d


def get_estimation_error(true_freq, estimated_freq, estimation_size):
    freq_err = 0
    for ip, freq in true_freq[:estimation_size]:
        if ip in list(zip(*estimated_freq[:estimation_size]))[0]:
            freq_err += abs(freq - dict(estimated_freq[:estimation_size])[ip])
        else:
            freq_err += freq

        return freq_err


def get_estimation_error_m(true_freq, estimated_freq, estimation_size):
    freq_err = 0

    for [freq, ip] in estimated_freq[:estimation_size]:
        if ip in true_freq:
            freq_err += abs(true_freq[ip] - freq)

    return freq_err


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
