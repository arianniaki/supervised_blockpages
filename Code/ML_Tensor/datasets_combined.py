"""
Arian Niaki
Dec 2017, UMass Amherst
This is the class for loading datasets of images and HTMLs
"""


import os
import glob
import numpy as np
import cv2
from sklearn.utils import shuffle
from bs4 import BeautifulSoup

def load_train(train_path, image_size, classes):
    docs = []
    images = []
    labels = []
    ids = []
    cls = []
    filename = 'glove.42B.300d.txt'

    embeddings_index = {}
    # f = open(filename, 'r')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    import pickle
    embeddings_index = {}
    with open('glove.pickle', 'rb') as handle:
        embeddings_index = pickle.load(handle)
    print('Found %s word vectors.' % len(embeddings_index))


    print('Reading training images')
    for fld in classes:   # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(train_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)


            htmlfile = fl.replace('png','html')
            f = open(htmlfile)
            doc = f.read()
            f.close()
            soup = BeautifulSoup(doc, "lxml")
            for script in soup(['script','style']):
                script.extract()
            # body = soup.find('body')
            body = soup.get_text()
            # print body
            if body is not None:
                body_text = body
                body_text = body_text.rstrip().lstrip()
                body_text_temp = body_text.replace('\n', '').replace('\t','')

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
                        word = selected_words[w].strip().lower().replace('.', '')
                    else:
                        word = selected_words[w]
                    # em = embeddings_index.get(selected_words[w])
                    em = embeddings_index.get(word)
                    if em is None:
                        em = np.zeros(300)
                    mat.append(np.array(em))

                for i in range(200 - len(mat)):
                    mat.append(np.zeros((300)))
                input = np.array(mat)
                # print input.shape
            docs.append(input)

    docs = np.array(docs)

    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)

    return images, docs, labels, ids, cls

def load_test_label(test_path, image_size, classes):
    filename = 'glove.42B.300d.txt'
    import pickle
    embeddings_index = {}
    with open('glove.pickle', 'rb') as handle:
        embeddings_index = pickle.load(handle)

    # f = open(filename, 'r')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     embeddings_index[word] = coefs
    # f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    docs = []

    images = []
    labels = []
    ids = []
    cls = []

    print('Reading testing images')
    for fld in classes:  # assuming data directory has a separate folder for each class, and that each folder is named after the class
        index = classes.index(fld)
        print('Loading {} files (Index: {})'.format(fld, index))
        path = os.path.join(test_path, fld, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            ids.append(flbase)
            cls.append(fld)


            htmlfile = fl.replace('png','html')

            f = open(htmlfile)
            doc = f.read()
            f.close()
            soup = BeautifulSoup(doc, "lxml")
            body = soup.find('body')
            if body is not None:
                body_text = body.text
                body_text = body_text.rstrip().lstrip()
                body_text_temp = body_text.replace('\n', '')
                body_text_temp2 = body_text_temp.replace('\t', '').replace('?',' ')
                # docs.append(body_text_temp2)
                words = body_text_temp2.split(' ')
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

                    # print selected_words
                    mat = []
                    for w in range(len(selected_words)):
                        import sys
                        reload(sys)
                        sys.setdefaultencoding('utf8')
                        if selected_words[w] != 0:
                            word = selected_words[w].strip().lower().replace('.', '')
                        else:
                            word = selected_words[w]
                        # em = embeddings_index.get(selected_words[w])
                        em = embeddings_index.get(word)
                        if em is None:
                            em = np.zeros(300)
                        mat.append(np.array(em))

                        # input = np.mean(np.asarray(mat), axis=0)
                        #                a = np.array(mat)
                    # print(a.shape, len(mat))

                    for i in range(200 - len(mat)):
                        mat.append(np.zeros((300)))
                    input = np.array(mat)

            else:
                input = np.zeros((200,300))
            # if body is None:
            #     print("BODY IS NONE for ", fl)
            #     input=(np.zeros(100))
            docs.append(input)

    docs = np.array(docs)
    images = np.array(images)
    labels = np.array(labels)
    ids = np.array(ids)
    cls = np.array(cls)
    return images, docs, labels, ids, cls

class DataSet(object):

  def __init__(self, images, docs, labels, ids, cls):
    """Construct a DataSet. one_hot arg is used only if fake_data is true."""

    self._num_examples = images.shape[0]

    docs = docs.astype(np.float32)

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    # Convert from [0, 255] -> [0.0, 1.0].

    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._docs = docs
    self._images = images
    self._labels = labels
    self._ids = ids
    self._cls = cls
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def docs(self):
    return self._docs

  @property
  def images(self):
    return self._images

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

    return self._images[start:end], self._docs[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size=0):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, docs, labels, ids, cls = load_train(train_path, image_size, classes)
  images, docs, labels, ids, cls = shuffle(images, docs, labels, ids, cls)  # shuffle the data
  print len(images)
  print len(docs)
  print len(ids)
  print len(labels)
  print len(cls)

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_docs = docs[:validation_size]

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_ids = ids[:validation_size]
  validation_cls = cls[:validation_size]

  train_docs = docs[validation_size:]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_ids = ids[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_docs, train_labels, train_ids, train_cls)
  data_sets.valid = DataSet(validation_images, validation_docs, validation_labels, validation_ids, validation_cls)

  return data_sets

def read_test_set_labeled(test_path, image_size, classes):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, docs, labels, ids, cls = load_test_label(test_path, image_size, classes)
    images, docs, labels, ids, cls = shuffle(images, docs, labels, ids, cls)  # shuffle the data

    test_docs = docs
    test_images = images
    test_labels = labels
    test_ids = ids
    test_cls = cls

    data_sets.test = DataSet(test_images, test_docs, test_labels, test_ids, test_cls)

    return data_sets
# def read_test_set(test_path, image_size):
#     images, ids  = load_test(test_path, image_size)
#     return images, ids

