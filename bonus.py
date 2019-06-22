import copy

# 'ip_src', 'duration', 'protocol', 'flags', 'packets', 'bytes'
def create_adversarial_examples(X_test, y_test, alter_packets, alter_bytes, alter_duration, logger):

  X_test_ = copy.deepcopy(X_test.tolist())
  if alter_packets != 0:
    X_test_ = [elem[:4] + [elem[4] + alter_packets] + elem[5:] for elem in X_test_]
  if alter_bytes != 0:
    X_test_ = [elem[:5] + [elem[5] + alter_bytes] for elem in X_test_]
  if alter_duration!= 0:
    X_test_ = [elem[:1] + [elem[1] + alter_duration] + elem[2:] for elem in X_test_]

  logger.log("Alter flows with {} packets, {} bytes, {} duration".format(
    alter_packets, alter_bytes, alter_duration))

  return X_test_