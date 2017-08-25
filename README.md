# MultinomialNB_DecisionTree_comparison

In questo esercizio si utilizzano implementazioni disponibili di Naive Bayes e Decision Tree
(scikit-learn) al ﬁne di confrontare le prestazioni di tali algoritmi su diversi data sets.
SOno stati scelti dei datasets fra il repository MLData e quelli integrati in scikit-learn,
ciascuno con dimensione n di almeno 1000 esempi. Si sono confrontate quindi le prestazioni dei
due algoritmi su ciascun dataset, misurando l’errore di generalizzazione al crescere del numero
di esempi. A tale scopo, è stata riportata su una "learning curve" la media e la deviazione standard
dell’errore sul test set, campionando un certo numero di training sets di dimensione m < n, facendo
variare n in scala logaritmica tra il 10% ed il 50% di n (riservando il restante 50% per il test set).
