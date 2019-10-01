#import io
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk import FreqDist
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from nltk import NaiveBayesClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from nltk import classify 

categories =['neg', 'pos']    


directory = 'rt-polaritydata/rt-polaritydata/training'
train_dirctory = 'rt-polaritydata/rt-polaritydata/testing'
reviews = load_files(directory)
review_test = load_files(train_dirctory)
docs_test = review_test.data

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(reviews.data)


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# training the classifier
#Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, reviews.target)
docs_new = [' this is a terrible move and I would never watch it']
X_new_counts = count_vect.transform(docs_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, reviews.target_names[category]))

#building a pipeline
text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
text_clf.fit(reviews.data,reviews.target)
#evaluation of model
predicted = text_clf.predict(docs_test)
print('Naive Bayes accuracy %r:' % np.mean(predicted == review_test.target))
print('Naive Bayes model confusion Matrix')
print(metrics.confusion_matrix(review_test.target, predicted))
#print(metrics.classification_report(reviews.target, predicted, target_names=review_test.target_names))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(reviews.data[:400], reviews.target[:400])
print(reviews.target_names[gs_clf.predict(['God is love'])[0]])
print(gs_clf.best_score_ )

for param_name in sorted(parameters.keys()):
   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# Support Vector Machine model
text_clf = Pipeline([('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',
alpha=1e-3, random_state=42,
max_iter=5, tol=None)),])
text_clf.fit(reviews.data, reviews.target)

#SVM evaluation
predicted = text_clf.predict(docs_test)
print('Support Vector Machine accuracy %r:' %np.mean(predicted == review_test.target) )
print('Support Vector Machin model confusion Matrix')
print(metrics.confusion_matrix(review_test.target, predicted))

#print(metrics.classification_report(reviews.target, predicted,target_names=review_test.target_names))

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(reviews.data[:400], reviews.target[:400])
print(reviews.target_names[gs_clf.predict(['God is love'])[0]])
print(gs_clf.best_score_ )

for param_name in sorted(parameters.keys()):
   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


#Logistic Regression model
pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', LogisticRegression())
])
pipeline.fit(reviews.data,reviews.target)
predicted = pipeline.predict(docs_test)

print('Logistic regression accuracy %r:' %np.mean(predicted == review_test.target) )

print('Logistic Regression confusion Matrix')
print(metrics.confusion_matrix(review_test.target, predicted))
#print(metrics.classification_report(reviews.target, predicted,target_names=review_test.target_names))

#LR evaluation
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}
gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
gs_clf = gs_clf.fit(reviews.data[:400], reviews.target[:400])
print(reviews.target_names[gs_clf.predict(['God is love'])[0]])
print(gs_clf.best_score_ )

for param_name in sorted(parameters.keys()):
   print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))