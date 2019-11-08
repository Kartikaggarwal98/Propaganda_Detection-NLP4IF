# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import os
import glob
import os.path
import sys

train_folder = "" # check that the path to the datasets folder is correct,
dev_folder = ""     # if not adjust these variables accordingly
train_labels_folder = ""
dev_template_labels_file = ""
dev_template_labels_file = ""

def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    
    file_list = glob.glob(os.path.join(folder_name, file_pattern))
    article_id_list, sentence_id_list, sentence_list = ([], [], [])
    for filename in sorted(file_list):
        article_id = os.path.basename(filename).split(".")[0][7:]
        with open(filename, "r", encoding="utf-8") as f:
            for sentence_id, row in enumerate(f.readlines(), 1):
                sentence_list.append(row.rstrip())
                article_id_list.append(article_id)
                sentence_id_list.append(str(sentence_id))

    return article_id_list, sentence_id_list, sentence_list


def are_ids_aligned(article_id_list, sentence_id_list,
                    reference_article_id_list, reference_sentence_id_list):
    """
    check whether the two lists of ids of the articles and the sentences are aligned
    """
    for art, ref_art, sent, ref_sent in zip(article_id_list, reference_article_id_list,
                                            sentence_id_list, reference_sentence_id_list):
        if art != ref_art:
            print("ERROR: article ids do not match: article id = %s, reference article id = %s"%(art, ref_art))
            return False
        if sent != ref_sent:
            print("ERROR: sentence ids do not match: article id:%s,%s sentence id:%s,%s" %(art, ref_art, sent, ref_sent))
            return False
    return True


def read_predictions_from_file(filename):
   
    articles_id, sentence_id_list, gold_labels = ([], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, sentence_id, gold_label = row.rstrip().split("\t")
            articles_id.append(article_id)
            sentence_id_list.append(sentence_id)
            gold_labels.append(gold_label)
    return articles_id, sentence_id_list, gold_labels


def read_predictions_from_file_list(folder_name, file_pattern):
    gold_file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles_id, sentence_id_list, gold_labels = ([], [], [])
    for filename in sorted(gold_file_list):
        art_ids, sent_ids, golds = read_predictions_from_file(filename)
        articles_id += art_ids
        sentence_id_list += sent_ids
        gold_labels += golds
    return articles_id, sentence_id_list, gold_labels

# loading articles' content from *.txt files in the train folder
train_article_ids, train_sentence_ids, sentence_list = read_articles_from_file_list(train_folder)

# loading gold labels, articles ids and sentence ids from files *.task-SLC.labels in the train labels folder
reference_articles_id, reference_sentence_id_list, gold_labels = read_predictions_from_file_list(
    train_labels_folder, "*.task-SLC.labels")



tdf=pd.DataFrame(list(zip(sentence_list,gold_labels)),columns=['sentence','labels'])
tdf.to_csv('/content/drive/My Drive/emnlp/datasets/traindf.csv')

len(train_article_ids),len(reference_articles_id)

# checking that the number of sentences in the raw training set and the gold label file
if not are_ids_aligned(train_article_ids, train_sentence_ids, reference_articles_id, reference_sentence_id_list):
    sys.exit("Exiting: training set article ids and gold labels are not aligned")
print("Loaded %d sentences from %d articles" % (len(sentence_list), len(set(train_article_ids))))

dev_article_id_list, dev_sentence_id_list, dev_sentence_list = read_articles_from_file_list(dev_folder)
reference_articles_id, reference_sentence_id_list, dev_labels = read_predictions_from_file(dev_template_labels_file)

len(dev_article_id_list),len(dev_sentence_list)

tdf=pd.DataFrame(list(zip(dev_article_id_list,dev_sentence_list)),columns=['id','sentence'])
tdf.to_csv('/content/drive/My Drive/emnlp/datasets/devdf.csv')

