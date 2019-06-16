import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def create_barplot(data_df, feature, plt_title, filename, logger):

  sns.set()
  fig = plt.figure(figsize = (20,10))
  
  x, y = zip(*Counter(data_df[feature]).items())
 
  plt.bar(x, y, color = "blue", log = True)
  
  plt.xlabel(feature, fontsize = 16)
  plt.ylabel("Count", fontsize = 16)
  plt.xticks(fontsize = 14, rotation = 90)
  plt.title(plt_title, fontsize = 16)

  plt.savefig(logger.get_output_file(filename), dpi = 120, bbox_inches='tight')


def create_categories_heatmap(data_df, feats_pair, filename, logger):

  selected_df = data_df[feats_pair + ['label']]

  selected_df_legit  = selected_df[selected_df.label == "LEGITIMATE"]
  selected_df_infect = selected_df[selected_df.label == "Botnet"]

  feats_pair_infect = selected_df_infect.groupby(feats_pair).agg('count')
  feats_pair_infect.reset_index(inplace = True)
  feats_pair_infect = feats_pair_infect.pivot(index = feats_pair[0], columns = feats_pair[1],
    values = 'label')
  feats_pair_infect.fillna(0, inplace = True)
  feats_pair_infect = feats_pair_infect.apply(lambda x: np.log(x + 1), axis = 1)

  feats_pair_legit = selected_df_legit.groupby(feats_pair).agg('count')
  feats_pair_legit.reset_index(inplace = True)
  feats_pair_legit = feats_pair_legit.pivot(index = feats_pair[0], columns = feats_pair[1],
    values = 'label')
  feats_pair_legit.fillna(0, inplace = True)
  feats_pair_legit = feats_pair_legit.apply(lambda x: np.log(x + 1), axis = 1)

  fig, ax = plt.subplots(1,2, figsize=(16,5))
  
  colormap = sns.light_palette("red", n_colors = 100, as_cmap = True)
  #sns.diverging_palette(220, 10, as_cmap=True)

  ax[0].title.set_text("Infected")
  sns.heatmap(feats_pair_infect, cmap = colormap, ax = ax[0])

  sns.heatmap(feats_pair_legit, cmap = colormap, ax = ax[1])
  ax[1].title.set_text("Legitimate")

  plt.savefig(logger.get_output_file(filename), dpi = 120, bbox_inches='tight')


def create_time_plot(df, df_host, feature, filename, logger):

  sns.set()
  fig = plt.figure(figsize=(22,9))

  ax = [None, None]
  ax[0] = plt.subplot(121)
  ax[0].title.set_text("Infected host")
  ax[0].title.set_fontsize(16)
  ax[0].set_ylabel(feature)
  ax[0].set_xlabel("Time")
  ax[0].ticklabel_format(style = "plain")
  plt.plot(df_host.date, df_host[feature])

  df_legit = df[df.label != "Botnet"]
  ax[1] = plt.subplot(122)
  ax[1].title.set_text("Normal hosts")
  ax[1].title.set_fontsize(16)
  ax[1].set_ylabel(feature)
  ax[1].set_xlabel("Time")
  ax[1].ticklabel_format(style = "plain")
  plt.plot(df_legit.date, df_legit[feature])
 
  plt.xticks(fontsize = 14)
  plt.yticks(fontsize = 14)

  plt.savefig(logger.get_output_file(filename), dpi = 120, bbox_inches='tight')


if __name__ == "__main__":
  print("Library module. Not main function.")