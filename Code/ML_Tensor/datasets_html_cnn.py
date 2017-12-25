"""
Arian Niaki
Dec 2017, UMass Amherst
This is the class for loading datasets of HTMLs
"""

import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
import time
from datetime import timedelta
import pickle
def load_train(train_path, classes):
    print ('loading GloVe matrix')
    start_time = time.time()
    filename = 'glove.42B.300d.txt'

    embeddings_index = {}

    with open('glove.pickle', 'rb') as handle:
        embeddings_index = pickle.load(handle)

    #f = open(filename, 'r')
    #for line in f:
    #   values = line.split()
    #    word = values[0]
    #    coefs = np.asarray(values[1:], dtype='float32')
    #    embeddings_index[word] = coefs
    #f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))

    docs = []
    labels = []
    ids = []
    cls = []

    print('Reading training htmls')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*html')

        files = glob.glob(path)
	len_files = len(files)
        for fl in files:
	    print(len_files)
	    len_files -= 1
            f = open(fl)
            doc = f.read()
            f.close()
            soup = BeautifulSoup(doc, "lxml")
            for script in soup(['script','style']):
                script.extract()
            # body = soup.find('body')
            body = soup.get_text()
	    del soup
            # print body
            # print fl
            if body is not None:
#                body_text = body
#                body_text = body.rstrip().lstrip()
                body_text_temp = body.rstrip().lstrip().replace('\n', '').replace('\t','')
                # body_text_temp2 = body_text_temp.replace('?',' ')
                # print body_text_temp
                # docs.append(body_text_temp2)
                words = body_text_temp.split(' ')
                # while u'' in words:
                #     words.remove(u'')
                selected_words = []
                if len(words) < 200:
                    # listofzeros = [0] * (1500 - len(words))
                    # selected_words = words + listofzeros
                    selected_words = words
                    if len(words)<2:
                        selected_words = ['somethingyouwouldntfindinembeddings']
                else:
                    selected_words = words[:200]

                # print selected_words
                mat = []
                for w in range(len(selected_words)):
                    import sys
                    reload(sys)
                    sys.setdefaultencoding('utf8')
                    if selected_words[w] != 0:
                        word = selected_words[w].strip().lower().replace('.','')
                    else:
                        word = selected_words[w]
                    # em = embeddings_index.get(selected_words[w])
                    em = embeddings_index.get(word)
                    if em is None:
                        em = np.zeros(300)
                    mat.append(np.array(em))

                #input = np.mean(np.asarray(mat), axis=0)
#                a = np.array(mat)
                # print(a.shape, len(mat))

                for i in range(200-len(mat)):
                    mat.append(np.zeros((300)))
                input = np.array(mat)
                # print input.shape
		del selected_words
		del mat	
		del em
            else:
                input = np.zeros((200,300))
            # print input[0]
            # if body is None:
            #     print("BODY IS NONE for ", fl)
            #     input=(np.zeros(100))
            docs.append(input)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    docs = np.array(docs)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    return docs, labels, ids, cls


def load_test_label(test_path, classes):
    filename = 'glove.42B.300d.txt'

    embeddings_index = {}
#    f = open(filename, 'r')
#    for line in f:
#        values = line.split()
#        word = values[0]
#        coefs = np.asarray(values[1:], dtype='float32')
#        embeddings_index[word] = coefs
#    f.close()

    embeddings_index = {}

    with open('glove.pickle', 'rb') as handle:
        embeddings_index = pickle.load(handle)
    print('Found %s word vectors.' % len(embeddings_index))

    docs = []
    labels = []
    ids = []
    cls = []

    print('Reading test htmls')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(test_path, fld, '*html')

        files = glob.glob(path)
	len_files = len(files)
        for fl in files:
	    print(len_files)
	    len_files -= 1
            f = open(fl)
            doc = f.read()
            f.close()
            soup = BeautifulSoup(doc, "lxml")
            for script in soup(['script','style']):
                script.extract()
            # body = soup.find('body')
            body = soup.get_text()
            # print body
            # print fl
            if body is not None:
                body_text = body
                body_text = body_text.rstrip().lstrip()
                body_text_temp = body_text.replace('\n', '').replace('\t','')
                # body_text_temp2 = body_text_temp.replace('?',' ')
                # print body_text_temp
                # docs.append(body_text_temp2)
                words = body_text_temp.split(' ')
                # while u'' in words:
                #     words.remove(u'')
                selected_words = []
                if len(words) < 200:
                    # listofzeros = [0] * (1500 - len(words))
                    # selected_words = words + listofzeros
                    selected_words = words
                    if len(words)<2:
                        selected_words = ['somethingyouwouldntfindinembeddings']
                else:
                    selected_words = words[:200]

                # print selected_words
                mat = []
                for w in range(len(selected_words)):
                    import sys
                    reload(sys)
                    sys.setdefaultencoding('utf8')
                    if selected_words[w] != 0:
                        word = selected_words[w].strip().lower().replace('.','')
                    else:
                        word = selected_words[w]
                    # em = embeddings_index.get(selected_words[w])
                    em = embeddings_index.get(word)
                    if em is None:
                        em = np.zeros(300)
                    mat.append(np.array(em))
                for i in range(200-len(mat)):
                    mat.append(np.zeros((300)))
                input = np.array(mat)
            else:
                input = np.zeros((200,300))
            # print input[0]
            # if body is None:
            #     print("BODY IS NONE for ", fl)
            #     input=(np.zeros(100))
            docs.append(input)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)
    docs = np.array(docs)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    return docs, labels, ids, cls
#    return docs[:-2], labels[:-2], ids[:-2], cls[:-2]  


# def load_test(test_path, image_size):
#   path = os.path.join(test_path, '*g')
#   files = sorted(glob.glob(path))
#
#   X_test = []
#   X_test_id = []
#   print("Reading test images")
#   for fl in files:
#       flbase = os.path.basename(fl)
#       img = cv2.imread(fl)
#       img = cv2.resize(img, (image_size, image_size), cv2.INTER_LINEAR)
#       X_test.append(img)
#       X_test_id.append(flbase)
#
#   ### because we're not creating a DataSet object for the test images, normalization happens here
#   X_test = np.array(X_test, dtype=np.uint8)
#   X_test = X_test.astype('float32')
#   X_test = X_test / 255
#
#   return X_test, X_test_id
#

class DataSet(object):

  def __init__(self, docs, labels, ids, cls):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    self._num_examples = docs.shape[0]


    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    docs = docs.astype(np.float32)
    # images = np.multiply(images, 1.0 / 255.0)

    self._docs = docs
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def docs(self):
    return self._docs

  @property
  def labels(self):
    return self._labels

  @property
  def ids(self):
    return self._ids

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1

      # # Shuffle the data (maybe)
      # perm = np.arange(self._num_examples)
      # np.random.shuffle(perm)
      # self._images = self._images[perm]
      # self._labels = self._labels[perm]
      # Start next epoch

      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._docs[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  docs, labels, ids, cls = load_train(train_path, classes)
  docs, labels, ids, cls = shuffle(docs, labels, ids, cls)  # shuffle the data

  if isinstance(validation_size, float):
    validation_size = int(validation_size * docs.shape[0])

  validation_docs = docs[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_docs = docs[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_docs, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_docs, validation_labels, validation_ids, validation_cls)

  return data_sets

def read_test_set_labeled(test_path, classes):
    class DataSets(object):
        pass

    data_sets = DataSets()

    docs, labels, ids, cls = load_test_label(test_path, classes)
    docs, labels, ids, cls = shuffle(docs, labels, ids, cls)  # shuffle the data

    test_docs = docs
    test_labels = labels
    test_ids = ids
    test_cls = cls

    data_sets.test = DataSet(test_docs, test_labels, test_ids, test_cls)

    return data_sets
# def read_test_set(test_path, image_size):
#     images, ids  = load_test(test_path, image_size)
#     return images, ids

