train_folder = "datasets/train-articles" # check that the path to the datasets folder is correct, 
dev_folder = "datasets/dev-articles"     # if not adjust these variables accordingly
train_labels_folder = "datasets/train-labels-SLC"
dev_template_labels_file = "datasets/dev.template-output-SLC.out"
task_SLC_output_file = "baseline-output-SLC.txt"

#
# Baseline for Task SLC
#
# Our baseline uses a logistic regression classifier on one feature only: the length of the sentence.
#
# Requirements: sklearn, numpy
#


from sklearn.linear_model import LogisticRegression
import glob
import os.path
import numpy as np
import sys


def read_articles_from_file_list(folder_name, file_pattern="*.txt"):
    """
    Read articles from files matching patterns <file_pattern> from  
    the directory <folder_name>. 
    The content of the article is saved in the array <sentence_list>.
    Each element of <sentence_list> is one line of the article.
    Two additional arrays are created: <sentence_id_list> and
    <article_id_list>, holding the id of the sentences and the article.
    The arrays <article_id_list> and <sentence_id_list> are the first
    two columns of the predictions for the article, i.e. the format
    of the file <dev_template_labels_file>, they will be used to match
    the sentences with their gold labels in <train_labels_folder> 
    or <dev_template_labels_file>.
    """
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
    """
    Reader for the gold file and the template output file. 
    Return values are three arrays with article ids, sentence ids and labels 
    (or ? in the case of a template file). For more info on the three 
    arrays see comments in function read_articles_from_file_list()
    """
    articles_id, sentence_id_list, gold_labels = ([], [], [])
    with open(filename, "r") as f:
        for row in f.readlines():
            article_id, sentence_id, gold_label = row.rstrip().split("\t")
            articles_id.append(article_id)
            sentence_id_list.append(sentence_id)
            gold_labels.append(gold_label)
    return articles_id, sentence_id_list, gold_labels


def read_predictions_from_file_list(folder_name, file_pattern):
    """
    Reader for the gold label files and the template output files
    <folder_name> is the folder hosting the files. 
    <file_pattern> values are {"*.task-SLC.labels", "*.task-SLC-template.out"}. 
    Return values are three arrays with article ids, sentence ids and labels 
    (or ? in the case of a template file). For more info on the three 
    arrays see comments in function read_articles_from_file_list()
    """
    gold_file_list = glob.glob(os.path.join(folder_name, file_pattern))
    articles_id, sentence_id_list, gold_labels = ([], [], [])
    for filename in sorted(gold_file_list):
        art_ids, sent_ids, golds = read_predictions_from_file(filename)
        articles_id += art_ids
        sentence_id_list += sent_ids
        gold_labels += golds
    return articles_id, sentence_id_list, gold_labels


### MAIN ###

# loading articles' content from *.txt files in the train folder
train_article_ids, train_sentence_ids, sentence_list = read_articles_from_file_list(train_folder)

# loading gold labels, articles ids and sentence ids from files *.task-SLC.labels in the train labels folder 
reference_articles_id, reference_sentence_id_list, gold_labels = read_predictions_from_file_list(train_labels_folder, "*.task-SLC.labels")

# checking that the number of sentences in the raw training set and the gold label file
if not are_ids_aligned(train_article_ids, train_sentence_ids, reference_articles_id, reference_sentence_id_list):
    sys.exit("Exiting: training set article ids and gold labels are not aligned")
print("Loaded %d sentences from %d articles" % (len(sentence_list), len(set(train_article_ids))))

# compute one feature for each sentence: the length of the sentence and train the model
train = np.array([ len(sentence) for sentence in sentence_list ]).reshape(-1, 1)
model = LogisticRegression(penalty='l2', class_weight='balanced', solver="lbfgs")
model.fit(train, gold_labels)

# reading data from the development set
dev_article_id_list, dev_sentence_id_list, dev_sentence_list = read_articles_from_file_list(dev_folder)
reference_articles_id, reference_sentence_id_list, dev_labels = read_predictions_from_file(dev_template_labels_file)
if not are_ids_aligned(dev_article_id_list, dev_sentence_id_list, reference_articles_id, reference_sentence_id_list):
    sys.exit("Exiting: development set article ids and gold labels are not aligned")

# computing the predictions on the development set
dev = np.array([ len(sentence) for sentence in dev_sentence_list ]).reshape(-1, 1)
predictions = model.predict(dev)

# writing predictions to file
with open(task_SLC_output_file, "w") as fout:
    for article_id, sentence_id, prediction in zip(dev_article_id_list, dev_sentence_id_list, predictions):
        fout.write("%s\t%s\t%s\n" % (article_id, sentence_id, prediction))
print("Predictions written to file " + task_SLC_output_file)
