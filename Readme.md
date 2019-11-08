# Fine-Grained Propaganda Detection
This repository contains the source code, dataset and the tools for EMNLP 2019 shared task.The task is part of the 2019 Workshop on NLP4IF: censorship, disinformation, and propaganda , co-located with the EMNLP-IJCNLP conference, November 3-7 2019, Hong Kong.
[Task Link](https://propaganda.qcri.org/nlp4if-shared-task/index.html)

The model is described in the paper https://www.aclweb.org/anthology/D19-5021.pdf

Citation
```
@inproceedings{aggarwal-sadana-2019-nsit,
    title = "{NSIT}@{NLP}4{IF}-2019: Propaganda Detection from News Articles using Transfer Learning",
    author = "aggarwal, Kartik  and
      Sadana, Anubhav",
    booktitle = "Proceedings of the Second Workshop on Natural Language Processing for Internet Freedom: Censorship, Disinformation, and Propaganda",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-5021",
    doi = "10.18653/v1/D19-5021",
    pages = "143--147",
    abstract = "In this paper, we describe our approach and system description for NLP4IF 2019 Workshop: Shared Task on Fine-Grained Propaganda Detection. Given a sentence from a news article, the task is to detect whether the sentence contains a propagandistic agenda or not. The main contribution of our work is to evaluate the effectiveness of various transfer learning approaches like ELMo, BERT, and RoBERTa for propaganda detection. We show the use of Document Embeddings on the top of Stacked Embeddings combined with LSTM for identification of propagandistic context in the sentence. We further provide analysis of these models to show the effect of oversampling on the provided dataset. In the final test-set evaluation, our system ranked 21st with F1-score of 0.43 in the SLC Task.",
}
```

## Task Description
The goal of the shared task is to produce models capable of spotting sentences and text fragments in which propaganda techniques are used in a news article.

We are  provided with a corpus of about 500 news articles in which specific propagandistic fragments have been manually spotted and labeled. 
