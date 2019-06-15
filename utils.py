import pandas as pd


def get_estimation_error(true_freq, estimated_freq, estimation_size):

  freq_err = 0

  for ip, freq in estimated_freq[:estimation_size]:
    if (ip, freq) in true_freq[:estimation_size]:
      freq_err += abs(dict(true_freq)[ip] - dict(estimated_freq)[ip])
    else:
      freq_err += dict(true_freq)[ip]

  return freq_err


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
      crt_content.append(line_list[0] + " " + line_list[1]) #date
      crt_content.append(line_list[2]) #duration
      crt_content.append(line_list[3]) #protocol
      crt_content.append(line_list[4].split(':')[0]) #ip_src
      crt_content.append(None if len(line_list[4].split(':')) == 1 else line_list[4].split(':')[1]) #port_src
      crt_content.append(line_list[6].split(':')[0]) #ip_dest
      crt_content.append(None if len(line_list[6].split(':')) == 1 else line_list[6].split(':')[1]) #port_dest
      crt_content.append(line_list[7]) #flags
      crt_content.append(line_list[8]) #tos
      crt_content.append(line_list[9]) #packets
      crt_content.append(line_list[10]) #bytes
      crt_content.append(line_list[11]) #flows
      crt_content.append(line_list[12]) #label

      file_content.append(crt_content)
        
      if line_idx % 100000 == 0:
        logger.log("Line {}: {}".format(line_idx, crt_content))

  logger.log("Finished loading file", show_time = True)

  return pd.DataFrame(file_content, columns = colnames)


if __name__ == "__main__":
  print("Library module. Not main function.")