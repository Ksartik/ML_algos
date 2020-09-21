import pandas as pd
import numpy as np
from collections import defaultdict
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectPercentile, chi2, f_classif
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import dill
import pickle
import json

nltk.download('stopwords')
nltk.download('punkt')
stemming = PorterStemmer()
stops = set(stopwords.words('english'))
tknzr = TweetTokenizer()


def ngramify(tokens, n):
    new_tokens = tokens
    if (n > 1):
        for i in range(len(tokens)-n+1):
            feat_n = ''
            for j in range(i, i+n):
                feat_n += tokens[j] + ' '
            new_tokens.append(feat_n[:-1])
    return new_tokens

def filtered_tokens(text):
    global stops, stemming, tknzr
    # tokens = nltk.word_tokenize(text)
    # tokens = text.split()
    tokens = tknzr.tokenize(text)
    # punctuation removed
    token_words = [stemming.stem(w) for w in tokens if (w.isalpha() and (not w in stops))]
    return token_words

def multinomial_event(probs):
    return list(probs.keys())[list(np.random.multinomial(1, list(probs.values())).ravel()).index(1)]

def accuracy (true_labels, pred_labels):
    n = pred_labels.shape[0]
    nacc_labels = 0
    for i in range(n):
        if (true_labels[i] == pred_labels[i]):
            nacc_labels += 1
    return nacc_labels/n

def confusion_mat (labels, true_labels, pred_labels):
    n = len(labels)
    confusion_mat = np.zeros(shape=(n, n))
    for i in range(true_labels.shape[0]):
        true_poli = labels.index(true_labels[i])
        pred_poli = labels.index(pred_labels[i])
        confusion_mat[pred_poli, true_poli] += 1
    return confusion_mat

class MI_Redundancy :
    def __init__ (self, sig_mi=0.1):
        self.word1 = defaultdict(int)
        self.word1_word2 = defaultdict(int)
        self.sig_mi = sig_mi

    def fit (self, text_data):
        for text in text_data:
            tokens = filtered_tokens(text)
            for token in tokens:
                self.word1[token] += 1
            for i in range(len(tokens)):
                for j in range(i+1, len(tokens)):
                    self.word1_word2[tokens[i] + '\t' + tokens[j]] += 1
    
    def calc_mi (self, word1, word2):
        total_word_pairs = np.sum(list(self.word1_word2.values()))
        try:
            xy = word1 + '\t' + word2
            return (np.log(self.word1_word2[xy]) + np.log(total_word_pairs) - np.log(self.word1[word1]) - np.log(self.word1[word2]))
        except:
            xy = word2 + '\t' + word1
            return (np.log(self.word1_word2[xy]) + np.log(total_word_pairs) - np.log(self.word1[word1]) - np.log(self.word1[word2]))
    
    def non_redundant_vocab (self):
        vocab = list(self.word1.keys())
        new_vocab = []
        for i in range(len(vocab)):
            for j in range(i+1, len(vocab)):
                if (abs(self.calc_mi(vocab[i], vocab[j])) < self.sig_mi):
                    new_vocab.append(vocab[i])
                    new_vocab.append(vocab[j])
                else:
                    new_vocab.append(vocab[i])
        return new_vocab

class TfidfGaussianNB:
    def __init__(self, nfeats=300, vocab=None):
        self.clf = GaussianNB()
        self.vectorizer = TfidfVectorizer(max_features=nfeats, dtype=np.float32, vocabulary=vocab)

    def train(self, train_data, train_labels, classes, feature_selection=False, percentile=100, batch_size=1000):
        if feature_selection:
            selector = SelectPercentile(chi2, percentile=percentile)
            X = selector.fit_transform(self.vectorizer.fit_transform(train_data), train_labels)
            new_vocab = list(np.array(self.vectorizer.vocabulary)[selector.get_support()])
            self.vectorizer = TfidfVectorizer(dtype=np.float32, vocabulary=new_vocab)
        print(len(self.vectorizer.vocabulary))
        for i in range(0, train_data.size, batch_size):
            print(i)
            data = train_data[i:i+batch_size]
            X = self.vectorizer.fit_transform(data).toarray()
            # self.clf.partial_fit(X, train_labels[i:i+batch_size], classes=classes)
            self.clf.partial_fit(X, train_labels[i:i+batch_size], classes=classes)

    def predict(self, data):
        return self.clf.predict(self.vectorizer.fit_transform(data).toarray())
    
    def load_model (self, filename):
        with open(filename + ".p", 'rb') as fp:
            self.clf = pickle.load(fp)

    def save_model (self, filename):
        pickle.dump(self.clf, open(filename + '.p', 'wb'))

def calculate_mi (n_x_y, n_y, n_x, n):
    return (np.log(n_x_y) - np.log(n_y) - np.log(n_x) + np.log(n))

def mi_word_class (nb_model, term, label):
    n_x = 0
    n = 0
    for lab in nb_model.labels:
        n_x += nb_model.wordsFreq_label[lab][term]
        n += nb_model.nwords_label[lab]
    n_y = nb_model.labels_freq[label]
    n_x_y = nb_model.wordsFreq_label[label][term]
    return calculate_mi(n_x_y, n_y, n_x, n)

class NaiveBayes :
    def __init__(self, ylabels, filtered_split = False, ngram = 1, laplace_c = 1, features=None, weights=None):
        self.labels_freq = {}
        self.nwords_label = {}
        self.wordsFreq_label = {}
        self.filtered_split = filtered_split
        self.ngram = ngram
        self.labels = ylabels
        self.laplace_c = laplace_c
        self.phi = {}
        self.phi_label = {}
        self.params = {}
        if (features is None):
            self.vocab = []
            self.fixed_features = False
            for label in ylabels:
                self.wordsFreq_label[label] = defaultdict(lambda:self.laplace_c)
                self.labels_freq[label] = 0
                self.nwords_label[label] = 0
            self.weights = None
        else:
            self.vocab = features
            self.fixed_features = True
            for label in ylabels:
                self.wordsFreq_label[label] = {}
                for feat in features:
                    self.wordsFreq_label[label][feat] = self.laplace_c
                self.labels_freq[label] = 0
                self.nwords_label[label] = 0
            self.weights = weights

    def calc_bayesian_params (self):
        m = len(self.labels)
        n = len(self.vocab)
        self.phi = {}
        for label in self.labels:
            self.phi[label] = defaultdict(lambda:self.laplace_c/self.nwords_label[label] if (self.nwords_label[label] != 0) else 0.0)
        for i in range(m):
            for j in range(n):
                label = self.labels[i]
                word = self.vocab[j]
                # self.phi[i][j] = self.wordsFreq_label[label][word]/self.nwords_label[label]
                self.phi[label][word] = self.wordsFreq_label[label][word]/self.nwords_label[label] if (self.nwords_label[label] != 0) else 0
    
    def calc_class_priors (self):
        for label in self.labels:
            self.phi_label[label] = self.labels_freq[label]
        total_freq = 0
        for label in self.labels:
            total_freq += self.labels_freq[label]
        for label in self.labels:
            self.phi_label[label] = self.phi_label[label]/total_freq

    def train (self, text_data, labels):
        for text, label in zip(text_data, labels):
            self.labels_freq[label] += 1
            if (not(self.filtered_split)):
                tokens = text.split()
            else:
                tokens = filtered_tokens(text)
            tokens = ngramify(tokens, self.ngram)
            for w in tokens:
                try: 
                    if (self.weights is not None):
                        self.wordsFreq_label[label][w] += 1 * self.weights[w]
                        self.nwords_label[label] += 1 * self.weights[w]
                    else:
                        self.wordsFreq_label[label][w] += 1
                        self.nwords_label[label] += 1
                except:
                    pass
        print("Loaded data")
        if (not(self.fixed_features)):
            vocab = set()
            for label in self.labels:
                vocab = vocab.union(set(list(self.wordsFreq_label[label].keys())))
            self.vocab = sorted(list(vocab))
            print("Vocabulary: ", str(len(self.vocab)))
        else:
            print("Vocabulary (fixed): ", str(len(self.vocab)))
        # Adding Laplace smoothing factor in the denominator : nword_label
        for label in self.labels:
            self.nwords_label[label] += self.laplace_c * len(self.vocab)
        self.calc_bayesian_params()
        self.calc_class_priors()
        self.params["phi"] = self.phi
        self.params["phi_label"] = self.phi_label

    def train_csv (self, train_file, cols, label_column, text_column):
        with open(train_file, "r", encoding="ISO-8859-1") as f:
            row_num = 0
            reader = csv.DictReader(f, cols)
            for row in reader:
                label = int(row[label_column])
                self.labels_freq[label] += 1
                if (not(self.filtered_split)):
                    tokens = row[text_column].split()
                else:
                    tokens = filtered_tokens(row[text_column])
                tokens = ngramify(tokens, self.ngram)
                for w in tokens:
                    try:
                        if (self.weights is not None):
                            self.wordsFreq_label[label][w] += 1 * self.weights[w]
                            self.nwords_label[label] += 1 * self.weights[w]
                        else:
                            self.wordsFreq_label[label][w] += 1
                            self.nwords_label[label] += 1
                    except:
                        pass
                row_num += 1
                # if (row_num % 1000 == 0):
                #     print(row_num)
        print("Loaded from file")
        if (not(self.fixed_features)):
            vocab = set()
            for label in self.labels:
                vocab = vocab.union(set(list(self.wordsFreq_label[label].keys())))
            self.vocab = sorted(list(vocab))
            print("Vocabulary: ", str(len(self.vocab)))
        else:
            print("Vocabulary (fixed): ", str(len(self.vocab)))
        # Adding Laplace smoothing factor in the denominator : nword_label
        for label in self.labels:
            self.nwords_label[label] += self.laplace_c * len(self.vocab)
        self.calc_bayesian_params()
        self.calc_class_priors()
        # return self.vocab, self.labels_freq, self.wordsFreq_label, self.nwords_label
        self.params["phi"] = self.phi
        self.params["phi_label"] = self.phi_label

    def prob_label_text (self, text):
        prob_label = {}
        prob_text_label = {}
        if (self.filtered_split):
            words = filtered_tokens(text)
        else:
            words = text.split()
        words = ngramify(words, self.ngram)
        for label in self.labels:
            prob_text_label[label] = 1.0
            for w in words:
                prob_text_label[label] *= self.phi[label][w]
        prob_text = 0.0
        for label in self.labels:
            prob_text += prob_text_label[label] * self.phi_label[label]
        for label in self.labels:
            prob_label[label] = (prob_text_label[label] * self.phi_label[label])/prob_text
        return prob_label
    
    def predict (self, test_data):
        calc_prob_label = np.vectorize(lambda text: self.prob_label_text(text))
        label_maxProb = np.vectorize(lambda d: max(d.keys(), key=(lambda k: d[k])))
        labels = label_maxProb(calc_prob_label(test_data))
        return labels
    
    def predict_prob (self, test_data):
        calc_prob_label = np.vectorize(lambda text: self.prob_label_text(text))
        return calc_prob_label(test_data)
    
    def load_model (self, filename):
        with open(filename + ".p", 'rb') as fp:
            # data = pickle.load(fp)
            data = json.load(fp)
        self.phi = data['phi']
        self.phi_label = data['phi_label']

    def save_model (self, filename):
        with open(filename + ".p", 'wb') as f:
            # pickle.dump(self.params, f, protocol=pickle.HIGHEST_PROTOCOL)
            json.dump(self.params, f)

