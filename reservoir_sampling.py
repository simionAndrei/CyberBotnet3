from queue import PriorityQueue
from tqdm import tqdm

import numpy as np

from logger import Logger

from utils import load_data

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

  print(reservoir_pq)
  reservoir = []
  while reservoir_pq.empty():
    reservoir.append(reservoir_pq.get(block = False))

  return reservoir


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  df = load_data("DATA_FILE1", logger)
  reservoir = reservoir_sample(df, 50, logger)