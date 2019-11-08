# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

!pip install flair
!pip install allennlp

import numpy as np
import pandas as pd
import os
import glob
import os.path
import sys

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from flair.datasets import ClassificationCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings,ELMoEmbeddings, BertEmbeddings,RoBERTaEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence


PATH=Path("/content/drive/My Drive/emnlp/")


df=pd.read_csv(PATH/'traindf.csv')

df=df.dropna() ## every alternate line is blank line

label_mapping={'propaganda':1,'non-propaganda':0} ## label encoding
df['labels']=df['labels'].apply(lambda x:label_mapping[x])

df=pd.concat([df,df[df['labels']==1].copy()],0).reset_index(drop=True) #oversampling

#convert to flair readable format
data=df[['labels','sentence']].rename(columns={'labels':"label", 'sentence':"text"})
data['label'] = '__label__' + data['label'].astype(str)

#train-test split
data.iloc[0:int(len(data)*0.8)].to_csv(PATH/'flair/train.csv', sep='\t', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*1)].to_csv(PATH/'flair/test.csv', sep='\t', index = False, header = False)


corpus = ClassificationCorpus(Path('/content/drive/My Drive/emnlp/flair/'), test_file='test.csv', dev_file='test.csv',train_file='train.csv')

print(corpus.obtain_statistics())


## use any pretrained stacked embedding from the FLAIR Framework
# embedding = RoBERTaEmbeddings()
# embedding = BertEmbeddings('bert-base-uncased')
embedding = ELMoEmbeddings('small')

#stack them with other embeddings
word_embeddings = [
            embedding,
#             FlairEmbeddings('news-forward',use_cache=True),
#             FlairEmbeddings('news-backward',use_cache=True),
        ]

#apply document LSTM to the stacked embeddings
document_embeddings = DocumentRNNEmbeddings(
        word_embeddings,
#         hidden_size=512,
#         reproject_words=True,
#         reproject_words_dimension=256,
    )

#build model
classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
trainer = ModelTrainer(classifier, corpus)

#specify parameters and train model
trainer.train(PATH/'models/', max_epochs=3,checkpoint=True, learning_rate=1e-1) 

classifier = TextClassifier.load('/content/drive/My Drive/emnlp/models/best-model.pt')



"""## Dev Set Prediction"""

dev_folder = ""     # if not adjust these variables accordingly
dev_template_labels_file = ""
task_SLC_output_file = ""



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

# reading data from the development set
dev_article_id_list, dev_sentence_id_list, dev_sentence_list = read_articles_from_file_list(dev_folder)

print (len(dev_article_id_list))


#start prediction

prd=[]
for x in range(len(dev)):
  print (x)
  prd.append(classifier.predict(Sentence(dev_sentence_list[x])))


preds=np.array([(int(y[0].labels[0].value) if len(y[0].labels) > 0 else 0) for y in prd])


# computing the predictions on the development set

label_inverse_mapping={1:'propaganda',0:'non-propaganda'} ## label encoding

predictions=np.array([label_inverse_mapping[x] for x in preds])


print (np.unique(predictions,return_counts=True),np.unique(preds,return_counts=True))

task_SLC_output_file = "/content/drive/My Drive/emnlp/submissions/finSub.txt"
# writing predictions to file
with open(task_SLC_output_file, "w") as fout:
    for article_id, sentence_id, prediction in zip(dev_article_id_list, dev_sentence_id_list, predictions):
        fout.write("%s\t%s\t%s\n" % (article_id, sentence_id, prediction))
print("Predictions written to file " + task_SLC_output_file)
