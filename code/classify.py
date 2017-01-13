from sklearn import tree
from itertools import izip
import pdb
#from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from  sklearn.naive_bayes import MultinomialNB
from  sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from collections import defaultdict
import argparse

human_bigram = defaultdict(int)
mt_bigram = defaultdict(int)

def general(lineA, lineB):
  ratio_char_count = len(lineA) / float(len(lineB))
  ratio_token_count = len(lineA.rstrip().split()) / float(len(lineB.strip().split()))
  ratio_mean_token_len = (len(lineA.rstrip())/ float(lineA.count(' ') + 1))  / (len(lineB.rstrip()) / float(lineB.count(' ') + 1))
  
  return [ratio_char_count, ratio_token_count, ratio_mean_token_len]
 

def token_match(lineA, lineB):
  '''
  Number of words in target sentence that appear in the source sentence
  These typically mean that the translator could not translate the word in the
  source sentence into the target language
  '''
  
  # Number of words in target sentence that appear in the source sentence
  not_translated_count = 0
  digit_count = 0
  source_set = set(lineA)
  for word in lineB:
    if not word.isdigit() and word in source_set:
      not_translated_count += 1
    elif word.isdigit():
      digit_count += 1

  # Ratio of words in target sentence that appear in the source sentence
  not_translated_ratio = 1
  if digit_count != len(lineB):
    not_translated_ratio = 1 - (not_translated_count / float(len(lineB) - digit_count))

  all_or_no_matches = 0
  if not_translated_count == 0 or not_translated_count == len(lineB):
    all_or_no_matches = 1
  return [not_translated_count, not_translated_ratio, all_or_no_matches]


def update_bigram(prev_source, curr_source, curr_target, bigram):
  if (prev_source, curr_source, curr_target) not in bigram:
    bigram[(prev_source, curr_source, curr_target)] = 0 
  bigram[(prev_source, curr_source, curr_target)] += 1 


def extract_features(source, target, train=True):
  fileA = open(source)
  fileB = open(target)
  features = []
  for lineA, lineB in izip(fileA, fileB):
    lineA = lineA[lineA.find(">")+1:lineA.find("</s")]
    lineB = lineB[lineB.find(">")+1:lineB.find("</s")]
    #print "%s\t%s" % (lineA.rstrip(), lineB.rstrip())
    #print len(lineA.rstrip().split()) 
    #print len(lineB.rstrip().split())

    token_features = token_match(lineA.rstrip().split(), lineB.rstrip().split())
    pair_features = []
    for feat in token_features:
      pair_features.append(feat)
  
    general_features = general(lineA, lineB)
    for feat in general_features:
      pair_features.append(feat)  


    if train:
      mt_bigram_count, human_bigram_count = 0, 0
      prev_source = '<s>' # place holder for begining of sentence
      for curr_source, curr_target in izip(lineA.rstrip().split(), lineB.rstrip().split()):
        if target.endswith("trans_ht"):
          human_bigram_count += 1
          update_bigram(prev_source, curr_source, curr_target, human_bigram) 
        elif target.endswith("trans_mt"):
          mt_bigram_count += 1
          update_bigram(prev_source, curr_source, curr_target, mt_bigram) 
        prev_source = curr_source  

      pair_features.append(human_bigram_count)
      pair_features.append(mt_bigram_count)
      #The ratio is mt_bigram_count : all bigram_counts
      if target.endswith("trans_ht"):
        pair_features.append(0)
      elif target.endswith("trans_mt"):
        pair_features.append(1)

    else:
      mt_bigram_count, human_bigram_count = 0, 0
      prev_source = '<s>' # place holder for begining of sentence
      for curr_source, curr_target in izip(lineA.rstrip().split(), lineB.rstrip().split()):
        if mt_bigram[(prev_source, curr_source, curr_target)] > human_bigram[(prev_source, curr_source, curr_target)]:
          mt_bigram_count += 1
        elif mt_bigram[(prev_source, curr_source, curr_target)] < human_bigram[(prev_source, curr_source, curr_target)]:
          human_bigram_count += 1
        prev_source = curr_source

      pair_features.append(human_bigram_count)
      pair_features.append(mt_bigram_count)
      if human_bigram_count == 0 and mt_bigram_count == 0:
        pair_features.append(.5)
      else:
        pair_features.append(mt_bigram_count / float(human_bigram_count + mt_bigram_count))

    features.append(pair_features)

  return features

def train(directory):
  print "train directory: " + directory
  #Extract Human Translation Features
  human_features = extract_features(directory + 'source_ht', directory + 'trans_ht', True)
  #Extract Machine Translation Features
  mt_features = extract_features(directory + 'source_mt', directory + 'trans_mt', True)
  return human_features, mt_features


def test(directory):
  print "test directory: " + directory
  human_features = extract_features(directory + 'source_ht', directory + 'trans_ht', False)
  mt_features = extract_features(directory + 'source_mt', directory + 'trans_mt', False)
  return human_features, mt_features

def main():

  PARSER = argparse.ArgumentParser(description="A classifier to predict whether a parallel corpus was generated by a human or machine translator")
  PARSER.add_argument("-tr", type=str, default="../clean_data/train/", help="directory containing training data")
  PARSER.add_argument("-te", type=str, default="../clean_data/dev/", help="directory containing test data")
  args = PARSER.parse_args()

  n_samples, n_features = [], []
  human_features, mt_features = train(args.tr)#args.s + "/train.txt", args.tm + "/train.txt", args.th + "/train.txt")


  #Add human_features to n_features
  for sent in human_features:
    n_samples.append(0)
    n_features.append(sent)

  #Add mt_features to n_features
  for sent in mt_features:
    n_samples.append(1)
    n_features.append(sent)

  #clf = KNeighborsClassifier(n_neighbors=3)

  #clf = BernoulliNB()
  #clf = MultinomialNB()
  #clf = GaussianNB()
  #clf = SGDClassifier(shuffle=True, loss="hinge", penalty="l2")
  #clf = svm.LinearSVC()
  clf = svm.SVC()
  #clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
  #clf = tree.DecisionTreeClassifier()
  clf = clf.fit(n_features, n_samples)

  test_human_features, test_mt_features = test(args.te)#args.s + "/dev.txt", args.tm + "/dev.txt", args.th + "/dev.txt")

  mt_correct, human_correct = 0, 0
  mt_wrong, human_wrong = 0, 0
  mt_total, human_total = 0, 0

  for test_mt in test_mt_features:
    if clf.predict(test_mt)[0] == 1:
      mt_correct += 1
    else:
      mt_wrong += 1
    mt_total += 1
    

  for test_human in test_human_features:
    if clf.predict(test_human)[0] == 0:
      human_correct += 1
    else:
      human_wrong += 1
    human_total += 1

  print "Accuracy: " + str((mt_correct + human_correct) / float(mt_total + human_total))
 
  print

  print "Human precission: " + str(human_correct / float(human_correct + mt_wrong)) 
  print "Human recall: " + str(human_correct / float(human_total))

  print

  print "MT precission: " + str(mt_correct / float(mt_correct + human_wrong)) 
  print "MT recall: " + str(mt_correct / float(mt_total))




if __name__ == '__main__':
  main()
