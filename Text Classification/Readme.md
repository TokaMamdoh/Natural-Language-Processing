# Text-classification
***
## 1 Overview
The main objective of this project is to build some machine learning models to classify text data and check their performance using many different packages and tools of python. 

## 2 Methodology
We followed some defined steps to obtain the aimed results:
#### 2.1 Import useful packages
NLTK, RegEx, pandas, and matplotlib among others, are crucial libraries for any text classification model.
#### 2.2 Pre-process the data
###### 2.2.1 Cleaning
Cleaning the data by removing all the special characters and stop words using RegEx library.
###### 2.2.2 Partitioning
Splitting the text data into words using RegEx library.
Splitting the whole text into smaller partitions, each partition holds 100 words.
###### 2.2.3 Label encoding
Convert the target variable (book’s name) from a categorical value to a numerical one.
#### 2.3 Feature Engineering
###### 2.3.1 Feature Extraction (Transformation)
Machines cannot process text data in its raw form. They need us to break down the text into a numerical format that’s easily readable, here comes the term of text transformation. In this assignment we used three techniques of text transformation which are BOW, TF-iDF, and n-grams.
###### 2.3.1.1 Bag of Words (BOW) or Count Vectorizer
A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms. The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents.
A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves two things:
1. vocabulary of known words.
2. measure of the presence of known words.
###### 2.3.1.2 Term Frequency-Inverse Document Frequency (TF-IDF)
  - Term Frequency (TF): is a measure of how frequently a term appears in a document.
  - Inverse Document Frequency (IDF): is a measure of how important this term is.
###### 2.3.1.3 N-Grams
We used n-grams as a parameter for BOW and TFIDF.
#### 2.4 Build the models
Building three models which are Support vector machine (SVM), K-nearest neighbors (KNN), and Decision tree (DT)

## 3 Result 
| Models | Accuracy with BOW | Accuracy with TF-IDF |
|---     |---                |---                   |
|SVM     |98.5 %             |99.5%                 |
|KNN     |86.5%              |96.5%                 |
|DT      |72%                |68.5%                 |

* So, the champion model is SVM with TFIDF
* Confusion matrix for SVM-TFIDF

![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/confusion%20matrix%20svm-tfidf.PNG)

## 4 Visualization
#### 4.1 Accuracies of all models 
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/Accuracies.PNG)
#### 4.2 Most common words of each books 
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/Gallipoli%20Diary%20book.PNG)
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/Greenmantle%20book.PNG)
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/Now%20It%20Can%20Be%20Told%20book.PNG)
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/The%20Four%20Horsemen%20of%20the%20Apocalypse%20book.PNG)
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/The%20Peace%20Negotiations%20book.PNG)
## 5 Error Analysis
the SVM-tfidf is performed very well, it misclassifyied for only one example that's because there is common word between the Gallipoli Diary book and the Greenmantle book
* the Word cloud graph show the words lead to misclassified
![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/Word%20cloud.PNG)
## 6 Decrease accuracy 
To need to decrease accuracy 20% to make the classification little bit harder to the model. to do that, we decrease the number of words in each example from 70 words to 10 words. So, the accutacy decreased from 99.5% to 70.5%
* Confusion matrix after decreasing accuracy

![](https://github.com/TokaMamdoh/Text-classification/blob/f093ae0b57d03850820c5fd5fcbe89987aa9b02d/images/confusion%20matrix%202.PNG)
