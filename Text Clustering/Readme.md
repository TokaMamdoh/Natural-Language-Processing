# Text Clustering 
***
## 1 Overview
We have many books from Gutenberg digital books, and we should take five different samples of these books, which are all of five different genres and of five different authors, that are semantically different. Then separate and set aside unbiased random partitions for training. The main objective is to produce similar clusters and compare them analyze the pros and cons of algorithms, generate and communicate the insights.

## 2 Methodology 
### 2.1 Cleaning 
cleaning the books by removing all letters except a to z and A to Z, and converting all characters into lowercase, then removing whitespace, and splitting the words by whitespace and removing all stop words and return the word into their original.
### 2.2 Partitions
Each book has 200 partitions and each partition consist of 100 words
### 2.3 Transformation
Using several types of transformations to choose the best from them.
#### 2.3.1 Bag of Word (BoW):
•   Bag-of-Words (BoW) works on assigning a unique number to each word and finding out the frequency of occurrence of this word in the text and converting the text int a fixed length vector format.
•	We use CountVectorizer() to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.
#### 2.3.2 Term Frequency- Inverse document frequency (TF-IDF):
•	Term Frequency (TF): is a measure of how frequently a term appears in a document.
•	Inverse Document Frequency (IDF): is a measure of how important this term is.
•	TF-IDF: is a method which gives us a numerical weightage of words which reflects how important the particular word is to a document in a corpus, a corpus is a collection of documents.
•	The main purpose of TF-IDF is intended to reflect how relevant a term is in a given document.
•	We use TfidfVectorizer() to help us in dealing with most frequent words. It converts a collection of raw documents to a matrix of TF-IDF features. Equivalent to CountVectorizer followed by TfidfTransformer. 
#### 2.3.3 Latent Dirichlet Allocation (LDA):
•	Latent Dirichlet Allocation (LDA): is a supervised dimensionality reduction technique. 
•	The goal of it is to project the features in higher dimensional space onto a lower-dimensional space in order to avoid the curse of dimensionality and also reduce resources and dimensional costs.
##### 2.3.3.1	Visualize the topics:
•	pyLDAvis: is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.
•	Gensim: is billed as a Natural Language Processing package that does 'Topic Modeling for Humans'. But it is practically much more than that. It is a leading and a state-of-the-art package for processing texts, working with word vector models (such as Word2Vec, FastText, etc.) and for building topic models.
#### 2.3.4 Word2Vec:
•	It represents words or phrases in vector space with several dimensions. Word embeddings can be generated using various methods like neural networks, co-occurrence matrix, probabilistic models, etc. Word2Vec consists of models for generating word embedding.
##### 2.3.4.1 Visualization for Word2Vec:
•	T-distributed Stochastic Neighbor Embedding (TSNE):  is a tool for visualizing high-dimensional data. T-SNE, based on stochastic neighbor embedding, is a nonlinear dimensionality reduction technique to visualize data in a two- or three-dimensional space.
### 2.4 Dimensionality Reduction:
•	Principal Component Analysis (PCA):  is a technique that comes from the field of linear algebra and can be used as a data preparation technique to create a projection of a dataset prior to fitting a model.
### 2.5 Clustering
Using several clustering algorithms and choose the best from them.
#### 2.5.1 K-Means clustering
•	K-means clustering: is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K.
#### 2.5.2 Expectation Maximization (EM)
• Expectation Maximization: The EM algorithm is an iterative approach that cycles between two modes. The first mode attempts to estimate the missing or latent variables, called the estimation-step or E-step. The second mode attempts to optimize the parameters of the model to best explain the data, called the maximization-step or M-step.
• E-Step: Estimate the missing variables in the dataset.
• M-Step: Maximize the parameters of the model in the presence of the data.

![](https://github.com/TokaMamdoh/Natural-Language-Processing/blob/c847c28efb291abf972763a848be56e7e3789dd8/Text%20Clustering/images/em11.jpg)
#### 2.5.3 Hierarchical
• Hierarchical clustering: is an algorithm that groups similar objects into groups called clusters. The endpoint is a set of clusters, where each cluster is distinct from each other cluster, and the objects within each cluster are broadly similar to each other.
• Hierarchical clustering starts by treating each observation as a separate cluster. Then, it repeatedly executes the following two steps: (1) identify the two clusters that are closest together, and (2) merge the two most similar clusters. This iterative process continues until all the clusters are merged together. This is illustrated in the diagrams below.

![](https://github.com/TokaMamdoh/Natural-Language-Processing/blob/132d362bec86ba7ef686f0de4f6a42456ae2a272/Text%20Clustering/images/Hierarchical-clustering-3.PNG)
![](https://github.com/TokaMamdoh/Natural-Language-Processing/blob/c847c28efb291abf972763a848be56e7e3789dd8/Text%20Clustering/images/Capture.PNG)
### 4 Evaluation
#### 4.1 Kappa 
Cohen's kappa coefficient (κ) is a statistic that is used to measure inter-rater reliability (and also intra-rater reliability) for qualitative (categorical) items.[1] It is generally thought to be a more robust measure than simple percent agreement calculation, as κ takes into account the possibility of the agreement occurring by chance. There is controversy surrounding Cohen's kappa due to the difficulty in interpreting indices of agreement. Some researchers have suggested that it is conceptually simpler to evaluate disagreement between items.

### 4.2 Silhouette
The silhouette coefficient is a measure of how similar a data point is within-cluster (cohesion) compared to other clusters (separation).





