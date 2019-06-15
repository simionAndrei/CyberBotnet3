from queue import PriorityQueue
from collections import Counter
from tqdm import tqdm

import numpy as np

from logger import Logger
from utils import load_data, get_estimation_error, plot_estimation_error

def true_count(data_df, logger):

  host_df = data_df[data_df['ip_src'] == logger.config_dict['INFECTED_HOST']]

  ip_occurences = Counter(host_df['ip_dest'].values).items()
  ip_occurences = sorted(ip_occurences, key=lambda x: x[1], reverse = True)

  return ip_occurences


def estimated_count(reservoir_list):

  return sorted(Counter(reservoir_list).items(), key=lambda x: x[1], reverse = True)


def reservoir_sample(data_df, reservoir_size, logger):

  reservoir_pq = PriorityQueue(maxsize = reservoir_size)

  for index, row in tqdm(data_df.iterrows()):

    if row['ip_src'] != logger.config_dict['INFECTED_HOST']:
      continue

    rank = np.random.uniform(low = 0.0, high = 1.0)
    if reservoir_pq.qsize() < reservoir_size:
      reservoir_pq.put((-rank, row['ip_dest']), block = False)
    else:
      crt_pair = reservoir_pq.get(block = False)
      reservoir_pq.put(crt_pair if -crt_pair[0] < rank else (-rank, row['ip_dest']))

  reservoir = []
  while not reservoir_pq.empty():
    reservoir.append(reservoir_pq.get(block = False)[1])

  return reservoir


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  df = load_data("DATA_FILE1", logger)

  reservoir_sizes = [50, 100, 200, 300, 500, 700, 900, 1000, 1100, 1200, 1300, 1443]
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
