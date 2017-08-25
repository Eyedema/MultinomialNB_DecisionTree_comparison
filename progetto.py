import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_digits
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets.mldata import fetch_mldata
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from variables import MY_ID, MY_KEY
import sys
import time

start_time = 0


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
    rng = np.random.RandomState(42)
    yy = []
    std_dev = []
    tries = 10
    for i in xx:
        yy_ = []
        for r in range(1, tries + 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, train_size=i, random_state=rng)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
        std_dev.append(np.std(yy))
    # transform yy and std_dev in numpy arrays (yy_, std_dev_) for the standard deviation plotting
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
    X_unscaled, y = mnist.data, mnist.target
    #   Scale data from [0, 255] to [0, 1] where value=1 iff old_value > 100.
    #   Since the data is from an image, we do not care about the grey area,
    #   we are just considering the white and black portion of the image.
    X = Binarizer(threshold=100).transform(X_unscaled)
    title = "Learning Curve Comparison for MNIST-binarized data"
    plot_learning_curve(X, y, title)


def leukemia_data():
    leukemia = fetch_mldata('leukemia')
    X_unscaled, y = leukemia.data, leukemia.target
    #   Scale data from [-n, m] to [0, 1]. The classificator can't handle
    #   negative data values.
    X = MinMaxScaler().fit_transform(X_unscaled)
    title = "Learning Curve Comparison for leukemia"
    plot_learning_curve(X, y, title)


def breast_cancer():
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    title = "Learning Curve Comparison for breast_cancer"
    plot_learning_curve(X, y, title)


def menu():
    print """Welcome,\n
            Please choose an option:\n
            1. 20newsgroups\n
            2. load_digits\n
            3. MNIST\n
            4. leukemia\n
            5. breast_cancer\n
            0. Quit"""
    choice = input(">>")
    exec_choice(choice)


def exec_choice(choice):
    global start_time
    if not isinstance(choice, int):
        print "Error"
        return
    try:
        start_time = time.time()
        actions[choice]()
    except KeyError:
        print "Invalid option"
        actions['menu']()


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
    'menu': menu,
    1: twenty_newsgroups,
    2: loaddigits,
    3: mnist_data,
    4: leukemia_data,
    5: breast_cancer,
    0: exit
}

menu()
