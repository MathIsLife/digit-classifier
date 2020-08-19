import gzip
import pickle
import numpy as np

def load_data():
  f = gzip.open('data/mnist.pkl.gz', 'rb')
  training_data, validation_data, test_data = pickle.load(f, encoding = 'bytes')
  f.close()
  return [training_data, validation_data, test_data]

def vectorized_result(x):
  ret = np.zeros(10)
  ret[x] = 1.0
  return ret

def load_data_wrapper():
  tr_d, va_d, te_d = load_data()
  training_inputs = [np.reshape(x, 784) for x in tr_d[0]]
  training_results = [vectorized_result(y) for y in tr_d[1]]
  training_data = zip(training_inputs, training_results)
  validation_inputs = [np.reshape(x, 784) for x in va_d[0]]
  validation_data = zip(validation_inputs, va_d[1])
  test_inputs = [np.reshape(x, 784) for x in te_d[0]]
  test_data = zip(test_inputs, te_d[1])
  return [training_data, validation_data, test_data]

