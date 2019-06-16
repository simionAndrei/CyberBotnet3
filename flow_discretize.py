from sklearn.cluster import KMeans

from logger import Logger
from utils import load_data

import matplotlib.pyplot as plt
import seaborn as sns

def elbow_plot(x, y, filename, logger):
  sns.set()
  fig = plt.figure(figsize=(8,6))

  plt.plot(x, y)
  plt.xticks([1, 3, 5, 7, 9, 10, 11, 12, 15, 17, 20])
  plt.xlabel("Number of clusters")
  plt.ylabel("Distance")
  plt.title("Sum of squared distances of samples to their closest cluster center")

  plt.savefig(logger.get_output_file(filename), dpi = 120, bbox_inches='tight')


def get_optimal_num_clusters(data_df, max_num_clusters, logger):

  numeric_data = data_df[['duration', 'packets', 'bytes']].values
  logger.log("Testing KMeans with {}-{} clusters".format(1, max_num_clusters))

  inertias = []
  for num_clusters in range(1, max_num_clusters):
    logger.log("Start KMeans with {} clusters".format(num_clusters))
    model = KMeans(n_clusters = num_clusters, max_iter = 1000, n_jobs = -1)
    model.fit(numeric_data)
    logger.log("Finished KMeans {}".format(num_clusters), show_time = True)

    inertias.append(model.inertia_)

  elbow_plot(range(1, max_num_clusters), inertias, "elbow_plot.png", logger)


def discretize(data_df, feature1, feature2, percentiles):

  feature1_space_size = data_df[feature1].unique().shape[0]
  feature2_space_size = data_df[feature2].unique().shape[0]

  space_size = feature1_space_size * feature2_space_size

  feature1_discrete_mapping = dict(zip(data_df[feature1].unique().tolist(), 
    range(feature1_space_size)))

  ordinal_ranks = {}
  for p in percentiles:
    ordinal_ranks[p] = int( (p/100) / feature2_space_size)

  feature2_discrete_mapping = {}
  sorted_feature2_space = sorted(data_df[feature2].unique().tolist())
  for elem in sorted_feature2_space:
    for i, p in enumerate(percentiles):
      if elem <= sorted_feature2_space[ordinal_ranks[p] - 1]:
        feature2_discrete_mapping[elem] = i
        break
    feature2_discrete_mapping[elem] = i

  codes = []
  for index, row in data_df.iterrows():
    code = feature1_discrete_mapping[row[feature1]] * (space_size / feature1_space_size) +\
           feature2_discrete_mapping[row[feature2]] * (space_size / feature1_space_size * feature2_space_size)
    codes.append(int(code))

  data_df[feature1 + "_" + feature2 + "_comb"] = codes


if __name__ == "__main__":

  logger = Logger(show = True, html_output = True, config_file = "config.txt")

  df = load_data("DATA_FILE2", logger)
  df = df[df.label != "Background"]

  #get_optimal_num_clusters(df, 20, logger)

  #percentiles computed by ELBOW using get_optimal_num_clusters(df, 20, logger)
  discretize(df, "flags", "bytes", percentiles = [20, 40, 60, 80])