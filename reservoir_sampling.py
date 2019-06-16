from queue import PriorityQueue
from collections import Counter
from tqdm import tqdm

import numpy as np

from logger import Logger
from utils import load_data, get_estimation_error, plot_estimation_error


def true_count(data_df, logger):

  host_df_out = data_df[data_df['ip_src'] == logger.config_dict['INFECTED_HOST1']]
  host_df_in  = data_df[data_df['ip_dest'] == logger.config_dict['INFECTED_HOST1']]

  ip_comb = Counter(np.concatenate((host_df_out['ip_dest'].values, host_df_in.['ip_src'].values)))
  ip_comb = sorted(ip_comb.items(), key=lambda x: x[1], reverse = True)

  return ip_comb


def estimated_count(reservoir_list):

  return sorted(Counter(reservoir_list).items(), key=lambda x: x[1], reverse = True)


def reservoir_sample(data_df, reservoir_size, logger):

  reservoir_pq = PriorityQueue(maxsize = reservoir_size)

  for index, row in tqdm(data_df.iterrows()):

    if row['ip_src'] == logger.config_dict['INFECTED_HOST1']:
      value = row['ip_dest']
    elif row['ip_dest'] == logger.config_dict['INFECTED_HOST1']:
      value = row['ip_src']
    else:
      continue

    rank = np.random.uniform(low = 0.0, high = 1.0)
    if reservoir_pq.qsize() < reservoir_size:
      reservoir_pq.put((-rank, value), block = False)
    else:
      crt_pair = reservoir_pq.get(block = False)
      reservoir_pq.put(crt_pair if -crt_pair[0] < rank else (-rank, value))

  reservoir = []
  while not reservoir_pq.empty():
    reservoir.append(reservoir_pq.get(block = False)[1])

  return reservoir


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  df = load_data("DATA_FILE1", logger)

  reservoir_sizes = [50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 5724]
  num_runs = 10
  top_n = 10
  avg_errors = []
  for reservoir_size in reservoir_sizes:
    errors = []
    for _ in num_runs:
      reservoir = reservoir_sample(df, reservoir_size, logger)

      true_ip_freq = true_count(df, logger)
      estimated_ip_freq = estimated_count(reservoir)

      crt_err = get_estimation_error(true_ip_freq, estimated_ip_freq, top_n)
      errors.append(crt_err)
    
    avg_errors.append(np.mean(errors))
    logger.log("Reservoir {}: mean error after {} runs {}".format(
      reservoir_size, num_runs, avg_errors[-1]), show_time = True)

  max_err = sum([elem[1] for elem in true_ip_freq[:top_n]])
  logger.log("Maximum possible estimation error {}".format(max_err))
  x = [top_n] + reservoir_sizes
  errors = [max_err] + avg_errors
  plot_estimation_error(x, errors, xlabel = "Reservoir size", 
    plt_title = "Reservoir sampling error at different reservoir sizes", 
    filename = "reservoir_sampling_err.png", logger = logger)
