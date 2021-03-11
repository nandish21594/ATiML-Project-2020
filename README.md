# ATiML-Project: Genre Identification on a subset of Gutenberg Corpus

### Abstract

This project is about Genre Identification on 996 books which are taken from Gutenberg Corpus. Corpus contains 9 genres. We are classifying each fiction book with a genre label not by using a bag of words model. First, features are extracted from each book using some popular libraries available for example, nltk, spacy etc and we have used three models and with the help of grid search, model selection is performed. Later test data is used to check how models are performing on unseen data and evaluation and visualization of results is performed.

Refer here for full paper: - [Genre Identification](https://github.com/nandish21594/genre_identification/blob/master/Project_Report.pdf)



### Steps for Execution

We have used Google Colab to extract features, run models, evaluate and vi- sualize. Some of the libraries for example: imblearn, textstat are not available in Colab, so we have installed them before running the code. For installing packages in Colab only pip can be used as Conda is not preinstalled in Colab. For installing any package we have used command like:

!pip install package name

- Go to directory Colab Notebooks to connect notebook with the drive • Import and Install packages
- Order of file execution:
  - Feature Extraction.ipynb
  - File final features without null values.csv is generated
  - Models Execution and Evaluation
    -  Naive Bayes Grid Search.ipynb
    - SVM on Original Features.ipynb
    - Random Forest Original Features.ipynb
    - BagofWords.ipynb
  - Dealing with Imbalanced Data.ipynb  
  - Gutenberg Visualization.ipynb
