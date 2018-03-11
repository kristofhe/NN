import tensorflow as tf

def main():
    filenames = ["C:/Users/Krist_000/Desktop/project_nn/R/d.txt"]
    dataset = tf.data.TextLineDataset(filenames)
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
                .skip(1)
                .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "\t"))))

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    session = tf.Session()
    session.run(dataset)
    print(dataset.output_classes)

'''
def input_fn(data_file, num_epochs, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have either run data_download.py or '
      'set both arguments --train_data and --test_data.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('income_bracket')
    return features, tf.equal(labels, '>50K')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

 # if shuffle:
 #   dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()
  return features, labels
  
  
  '''
if __name__ == '__main__':
    filenames = "C:/Users/Krist_000/Desktop/project_nn/R/d.txt"
    dataset = tf.data.TextLineDataset(filenames)