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

from variables import MY_ID, MY_KEY


def plot_learning_curve(X, y, title):
    classifiers = [
        ("Decision Tree", DecisionTreeClassifier(), (0, 0.627, 0.690)),
        ("Naive Bayes", MultinomialNB(), (0.921, 0.407, 0.254)),
        #   max depth = 8
    ]
    #   debug save image
    tries = 0
    for name, classifier, color in classifiers:
        xx, yy, std_dev, yy_, std_dev_, tries = compute_scores(X, y, classifier)
        plt.plot(xx, yy, 'o-', label=name, color=color)
        plt.fill_between(xx, yy_ - std_dev_,
                         yy_ + std_dev_, alpha=0.2, color=color)
    plt.title(title)
    plt.legend()
    plt.xlabel("Train size %")
    plt.ylabel("Test set error")
    plt.grid()
    #   save image debug
    save(title.split()[len(title.split()) - 1], " ".join(np.vectorize("%.2f".__mod__)(xx)), tries)
    plt.show()


#   http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#
def compute_scores(X, y, classifier):
    heldout = [.9, .8, .7, .6, .5]
    xx = 1. - np.array(heldout)
    yy = []
    std_dev = []
    tries = 10
    for i in xx:
        yy_ = []
        for r in range(1, tries + 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, train_size=i)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            yy_.append(1 - accuracy_score(y_true=y_test, y_pred=y_pred))
        yy.append(np.mean(yy_, dtype=np.float64))
        std_dev.append(np.std(yy_, dtype=np.float64))
    #   transform yy and std_dev in numpy arrays (yy_, std_dev_) for the standard deviation plotting
    #   and return them, along with the train percentages (xx), scores (yy), and std_dev.

    #   Return also tries-1 for debug to save the image
    yy_ = np.array(yy)
    std_dev_ = np.array(std_dev)
    return xx, yy, std_dev, yy_, std_dev_, tries


def twenty_newsgroups():
    newsgroups = fetch_20newsgroups()
    X = TfidfTransformer().fit_transform(CountVectorizer().fit_transform(newsgroups.data))
    y = newsgroups.target
    title = "Learning Curve Comparison for 20newsgroups"
    plot_learning_curve(X, y, title)


def loaddigits():
    digits = load_digits()
    X, y = digits.data, digits.target
    title = "Learning Curve Comparison for load_digits"
    plot_learning_curve(X, y, title)


def mnist_data():
    mnist = fetch_mldata('MNIST original')
    X, y = mnist.data, mnist.target
    title = "Learning Curve Comparison for MNIST"
    plot_learning_curve(X, y, title)


#   https://stackoverflow.com/questions/1270951/how-to-refer-to-relative-paths-of-resources-when-working-with-a-code-repository
def yahoo():
    fn = os.path.join(os.path.dirname(__file__), "yahoo-web-directory-topics.libsvm")
    X, y = load_svmlight_file(fn)
    title = "Learning Curve Comparison for Yahoo"
    print X.shape, y.shape
    plot_learning_curve(X, y, title)


def dmoz():
    fn = os.path.join(os.path.dirname(__file__), "dmoz-web-directory-topics.libsvm")
    X, y = load_svmlight_file(fn)
    title = "Learning Curve Comparison for DMOZ"
    print X.shape, y.shape
    plot_learning_curve(X, y, title)


def menu():
    print """Welcome,\n
            Please choose an option:\n
            1. 20newsgroups\n
            2. load_digits\n
            3. MNIST\n
            4. DMOZ\n
            5. yahoo\n
            0. Quit"""
    choice = input(">>")
    exec_choice(choice)


def exec_choice(choice):
    if not isinstance(choice, int):
        print "Error"
        return
    try:
        start_time = time.time()
        actions[choice]()
    except KeyError:
        print "Invalid option"
        menu()


def exit_program():
    sys.exit()


def save(name, percentages, tries):
    import telepot
    bot = telepot.Bot(MY_KEY)
    import datetime
    timestamp = time.time()
    message = "Running time for %s with %s tries and train percentages of" \
              " %s:\n%s seconds" % (name, tries, percentages, timestamp - start_time)
    print message
    value = datetime.datetime.fromtimestamp(timestamp)
    date = (value.strftime('%d-%m h%Hm%Ms%S'))
    name = 'plots/{} {} tries={} percentage={}.png'.format(name, date, tries, percentages)
    plt.savefig(name)
    bot.sendMessage(MY_ID, message)
    bot.sendPhoto(MY_ID, open(name, 'rb'))


actions = {
    1: twenty_newsgroups,
    2: loaddigits,
    3: mnist_data,
    4: dmoz,
    5: yahoo,
    0: exit
}

menu()
