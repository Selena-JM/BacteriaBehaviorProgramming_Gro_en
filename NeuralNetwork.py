import os
os.environ['TF_ENABLE_ONEDNN_OPTS']="0"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


### Data creation 
## Naive approach
X_naive = [[0, 0, 0, 0],
    [1,	0,	0,	0],
    [0,	1,	0,	0],
    [0,	0,	1,	0],
    [0,	0,	0,	1],
    [1,	1,	0,	0],
    [1,	0,	1,	0],
    [1,	0,	0,	1],
    [0,	1,	1,	0],
    [0,	1,	0,	1],
    [0,	0,	1,	1],
    [1,	1,	1,	0],
    [1,	1,	0,	1],
    [1,	0,	1,	1],
    [0,	1,	1,	1],
    [1,	1,	1,	1]]

# If we consider the activation of fluorescence 
Y_naive = [[0,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]

# If we consider a classification
Y_naive_classif = [[1,0,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,0,1]]

# If we consider categories
Y_naive_categories = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]

#Definition of the train and test sets
X_naive_train, X_naive_test, Y_naive_classif_train, Y_naive_classif_test = train_test_split(X_naive, Y_naive_classif, test_size=0.33, random_state=42)

## Approach based on linear regression : each variable is linearly created with respect to the risk, 
## with a director coefficient of 4. This method is not viable because each variable can be used alone for the 
## classification : if urea is greater than 0.85 for example it means it is a risk 4, which is false
# With this technique I don't need the data synthesis ? 
rows = 40
lim = int(np.floor(rows/5))
X_lin = np.zeros((rows,4))

# If we consider categories
Y_lin_categories = np.zeros((rows,1))
for i in range(lim) :
    Y_lin_categories[i] = 0
for i in range(lim, 2*lim) :
    Y_lin_categories[i] = 1
for i in range(2*lim, 3*lim) :
    Y_lin_categories[i] = 2
for i in range(3*lim, 4*lim) :
    Y_lin_categories[i] = 3
for i in range(4*lim, 5*lim) :
    Y_lin_categories[i] = 4

for i in range(rows) :
    X_lin[i][0] = abs(Y_lin_categories[i]/4 + (np.random.choice([-1,1]) * (np.random.random()-1)*10**(-1)))
    X_lin[i][1] = abs(Y_lin_categories[i]/4 + (np.random.choice([-1,1]) * (np.random.random()-1)*10**(-1)))
    X_lin[i][2] = abs(Y_lin_categories[i]/4 + (np.random.choice([-1,1]) * (np.random.random()-1)*10**(-1)))
    X_lin[i][3] = abs(Y_lin_categories[i]/4 + (np.random.choice([-1,1]) * (np.random.random()-1)*10**(-1)))

# If we consider a classification
Y_lin_classif = np.zeros((np.size(Y_lin_categories),5))
for i in range (np.size(Y_lin_categories)):
    if Y_lin_categories[i] == 0:
       Y_lin_classif[i] = [1,0,0,0,0]
    if Y_lin_categories[i] == 1:
        Y_lin_classif[i] = [0,1,0,0,0]
    if Y_lin_categories[i] == 2:
        Y_lin_classif[i] = [0,0,1,0,0]
    if Y_lin_categories[i] == 3:
        Y_lin_classif[i] = [0,0,0,1,0]
    if Y_lin_categories[i] == 4:
        Y_lin_classif[i] = [0,0,0,0,1]

#Definition of the train and test sets
X_lin_train, X_lin_test, Y_lin_classif_train, Y_lin_classif_test = train_test_split(X_lin, Y_lin_classif, test_size=0.33, random_state=42)
X_lin_train_cat, X_lin_test_cat, Y_lin_categories_train, Y_lin_categories_test = train_test_split(X_lin, Y_lin_categories, test_size=0.33, random_state=42)

## Approach based on data synthesis


### Decision tree 
# Of course this method gives perfect results because the training table is complete and gives all the 
# possible input-output

#function to extract human readable rules 
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

def decision_tree_method(X_train, Y_train, X_test, Y_test, cat=True,show=False):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, Y_train)


    Y_pred_tree = clf.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred_tree))

    feature_names = ["AI-2","Urea","Toxins","Siderophores"]
    if not cat:
        class_names = ["0","1","2","3"]
    else : 
        class_names = ["0","1","2","3","4"]


    text_representation = tree.export_text(clf, feature_names=feature_names)
    print(text_representation)

    rules = get_rules(clf, feature_names, class_names)
    for r in rules:
        print(r)
    
    if show:
        fig = plt.figure(figsize=(15,10))
        _ = tree.plot_tree(clf, 
                        feature_names=["AI-2","Urea","Toxins","Siderophores"],  
                        class_names=class_names,
                        filled=True)
        plt.show()


# decision_tree_method(X_lin_train_cat, Y_lin_categories_train, X_lin_test_cat, Y_lin_categories_test, cat=True, show=True)

### Linear equations 
def linear_eq_solving_method(X, Y):
    A, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    print(A)

    print(np.dot(A,np.reshape(X[5], (4,1)))) #Not satisfactory
    print(residuals)

# method does not give satisfactory results with vectors
A=[[0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05]]

# of course with categories it works perfectly with A = [1. 1. 1. 1.]
#linear_eq_solving_method(X, Y)

### Linear regression
# of course it works perfectly with categories : coeffs = [1,1,1,1] because when there is no input, y = 0,
# when there is one input y = 1 etc 
# Doesn't work with vectors though

def linear_reg_method(X, X_test, Y_cat,Y_test_cat):
    mlr = LinearRegression()  
    mlr.fit(X, Y_cat)
    print(mlr.coef_)


    print("Intercept: ", mlr.intercept_)
    print("Coefficients:")
    list(zip(X, mlr.coef_))

    y_pred_mlr= mlr.predict(X_test)
    print("Prediction for test set: {}".format(y_pred_mlr))
    print("Real values: {}".format(Y_test_cat))


    from sklearn import metrics
    meanAbErr = metrics.mean_absolute_error(Y_test_cat, y_pred_mlr)
    meanSqErr = metrics.mean_squared_error(Y_test_cat, y_pred_mlr)
    rootMeanSqErr = np.sqrt(metrics.mean_squared_error(Y_test_cat, y_pred_mlr))
    print('R squared: {:.2f}'.format(mlr.score(X,Y_cat)*100))
    print('Mean Absolute Error:', meanAbErr)
    print('Mean Square Error:', meanSqErr)
    print('Root Mean Square Error:', rootMeanSqErr)

### Neural network model
# see page 4 of the original paper for the different parameters 
# Here we want each category to be mutually incompatible so we add a layer with one neuron that has softmax activation function and 
def neural_network_method(X_lin_train,Y_lin_classif_train):
    model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=[4]),
                            keras.layers.Dense(5, activation='softmax')])   #keras.layers.Dense(4, activation='relu', input_shape=[4], kernel_regularizer=regularizers.l2(0.0001)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.1), metrics=['categorical_accuracy']) # 'categorical_crossentropy'
    model.fit(X_lin_train,Y_lin_classif_train)

    model.evaluate(X_lin_test,Y_lin_classif_test)
    Y_test_pred = model.predict(X_lin_test)
    # print(Y_test_pred)
    # Y_pred = model.predict(X)
    # print(Y_pred)
    Y_pred = np.argmax(Y_test_pred, axis=-1)
    print(Y_pred)

    first_layer_weights = model.layers[0].get_weights()[0]
    print(first_layer_weights)
# neural_network_method(X_lin_train,Y_lin_classif_train)


