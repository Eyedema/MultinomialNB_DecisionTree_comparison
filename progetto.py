import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_digits
from sklearn.datasets import load_svmlight_file
from sklearn.datasets.mldata import fetch_mldata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def plot_learning_curve(X, y, tries, title):
    """This function plots the datas once the compute_scores function has done evaluating them.

    Parameters
    -------
    X : the data portion of the dataset
    y : the target portion of the dataset
    tries: # of tries to be performed
    title: the plot title
    """
    global xx
    classifiers = [
        ("Decision Tree", DecisionTreeClassifier(), (0, 0.627, 0.690)),
        ("Naive Bayes", MultinomialNB(), (0.921, 0.407, 0.254)),
    ]
    for name, classifier, color in classifiers:
        xx, yy, std_dev, yy_, std_dev_ = compute_scores(X, y, tries, classifier)
        plt.plot(xx, yy, 'o-', label=name, color=color)
        plt.fill_between(xx, yy_ - std_dev_,
                         yy_ + std_dev_, alpha=0.2, color=color)
    plt.title(title)
    plt.legend()
    plt.xlabel("Train size %")
    plt.ylabel("Test set error")
    plt.grid()
    plt.show()


def compute_scores(X, y, tries, classifier):
    """This function evaluates the error on the test set and returns its mean and
    standard deviation on a fixed number of tries.

    This function is a modification of an example in the scikit-learn documentation
    that can be found on the following link:
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#


    Parameters
    -------
    X : the data portion of the dataset
    y : the target portion of the dataset
    tries: # of tries to be performed
    classifier: the classifier object, can be either a Naive Bayes or Decision Tree type object

    Returns
    -------
    tuple
        This function returns a tuple containing the train percentages from 10% to 50% in logarithmic scale,
        the mean of the error on the test set and the standard deviation on the same error. The function
        also returns a copy of this last two objects, but converted into a Numpy array to draw them.

    """
    heldout = [.9, .8, .7, .6, .5]
    xx = 1. - np.array(heldout)
    yy = []
    std_dev = []
    print "Training " + str(classifier) + "..."
    for i in xx:
        print str(i * 100) + "%"
        yy_ = []
        for r in range(1, tries + 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, train_size=i)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            yy_.append(1 - accuracy_score(y_true=y_test, y_pred=y_pred))
        yy.append(np.mean(yy_, dtype=np.float64))
        std_dev.append(np.std(yy_, dtype=np.float64))
    #   transform yy and std_dev in numpy arrays (yy_, std_dev_) for the standard deviation plotting
    #   and return them, along with the train percentages (xx), errors (yy), and std_dev.
    yy_ = np.array(yy)
    std_dev_ = np.array(std_dev)
    return xx, yy, std_dev, yy_, std_dev_


def twenty_newsgroups(tries):
    """This function loads the 20NewsGroups dataset integrated in scikit-learn and splits
    the data and the target.
    http://scikit-learn.org/stable/datasets/twenty_newsgroups.html
    """
    stop_words = select_stop_words()
    newsgroups = fetch_20newsgroups(subset="all", remove=("headers", "quotes", "footers"))
    X = TfidfTransformer().fit_transform(CountVectorizer(stop_words=stop_words).fit_transform(newsgroups.data))
    y = newsgroups.target
    title = "Learning Curve Comparison for 20newsgroups"
    plot_learning_curve(X, y, tries, title)


def loaddigits(tries):
    """This function loads the Digits dataset integrated in scikit-learn and splits the
    data and the target.
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    """
    digits = load_digits()
    X, y = digits.data, digits.target
    title = "Learning Curve Comparison for Digits"
    plot_learning_curve(X, y, tries, title)


def mnist_data(tries):
    """This function fetches the MNIST dataset from MLDATA and splits the data and the
    target.
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_mldata.html
    http://mldata.org/repository/data/viewslug/mnist/
    """
    mnist = fetch_mldata('mnist original')
    X, y = mnist.data, mnist.target
    title = "Learning Curve Comparison for MNIST"
    plot_learning_curve(X, y, tries, title)


#   https://stackoverflow.com/a/1270970/4014928
def yahoo(tries):
    """This function searches for the libsvm file containing the datasets and splits
    the data and the target.
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    http://mldata.org/repository/data/viewslug/yahoo-web-directory-topics/
    """
    fn = os.path.join(os.path.dirname(__file__), "yahoo-web-directory-topics.libsvm")
    X, y = load_svmlight_file(fn)
    title = "Learning Curve Comparison for Yahoo"
    plot_learning_curve(X, y, tries, title)


def dmoz(tries):
    """This function searches for the libsvm file containing the datasets and splits
    the data and the target.
    http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_svmlight_file.html
    http://www.mldata.org/repository/data/viewslug/dmoz-web-directory-topics/
    """
    fn = os.path.join(os.path.dirname(__file__), "dmoz-web-directory-topics.libsvm")
    X, y = load_svmlight_file(fn)
    title = "Learning Curve Comparison for DMOZ"
    plot_learning_curve(X, y, tries, title)


def menu():
    """This function prints the menu.
    :raise KeyError
            if the input is not int"""
    print """Welcome,\n
            Please choose an option:\n
            1. 20newsgroups\n
            2. load_digits\n
            3. MNIST\n
            4. DMOZ\n
            5. yahoo\n
            0. Quit"""
    try:
        choice = input(">>")
    except NameError:
        print "Error"
        menu()
    exec_choice(choice)


def exec_choice(choice):
    """This function sets up the # of tries and calls the right function to start the test.
    """
    global start_time
    if choice == 0:
        actions[choice]()
    print "How many tries would you like to perform?"
    try:
        tries = input(">>")
    except NameError:
        print "Error"
        menu()
    start_time = time.time()
    actions[choice](tries)


def select_stop_words():
    print """"Would you like to use the English dictionary for the stopwords?\n
          1. yes\n
          0. no """
    try:
        stopwords = input(">>")
    except NameError:
        print "Error"
        menu()
    if stopwords == 1:
        stop_words = 'english'
    else:
        stop_words = None
    return stop_words


def exit_program():
    """This function exits the program."""
    sys.exit()


actions = {
    1: twenty_newsgroups,
    2: loaddigits,
    3: mnist_data,
    4: dmoz,
    5: yahoo,
    0: exit_program
}

menu()
