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


### Data creation 
X = [[0, 0, 0,	0],
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

Y_cat = [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]

Y = [[0,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [1,0,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,0,1]]
Ybis = Y = [[1,0,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,1,0], [0,0,0,0,1]]

X_test = [X[i] for i in [0,5,9,15]]

Y_test = [Y[i] for i in [0,5,9,15]]
Y_test_bis = [Ybis[i] for i in [0,5,9,15]]


### Trying with decision trees 
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

def decision_tree_method(X, Y, cat=True,show=False):
    clf = tree.DecisionTreeClassifier(random_state=0)
    clf.fit(X, Y)


    Y_pred_tree = clf.predict(X)
    print("Accuracy:", accuracy_score(Y, Y_pred_tree))

    if not cat:
        class_names = ["0","1","2","3"]
    else : 
        class_names = ["0","1","2","3","4"]

    if show:
        feature_names = ["AI-2","Urea","Toxins","Siderophores"]
        fig = plt.figure(figsize=(15,10))
        _ = tree.plot_tree(clf, 
                        feature_names=["AI-2","Urea","Toxins","Siderophores"],  
                        class_names=class_names,
                        filled=True)
        plt.show()

    text_representation = tree.export_text(clf, feature_names=feature_names)
    print(text_representation)

    rules = get_rules(clf, feature_names, class_names)
    for r in rules:
        print(r)


decision_tree_method(X, Y_cat, cat=True, show=False)

### Trying to solve the linear equations 
def linear_eq_solving_method(X, Y):
    A, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    print(A)

    print(np.dot(A,np.reshape(X[5], (4,1)))) #Not satisfactory

# method does not give satisfactory results
A=[[0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05],
    [0.05, 0.15, 0.15, 0.05]]



### Trying with a neural network model
# see page 4 of the original paper for the different parameters 
# Here we want each category to be mutually incompatible so we add a layer with one neuron that has softmax activation function and 

model = keras.Sequential([keras.layers.Dense(32, activation='relu', input_shape=[4]),
                          keras.layers.Dense(5, activation='softmax')])   #keras.layers.Dense(4, activation='relu', input_shape=[4], kernel_regularizer=regularizers.l2(0.0001)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.1), metrics=['categorical_accuracy']) # 'categorical_crossentropy'
model.fit(X,Ybis)

model.evaluate(X_test,Y_test_bis)
Y_test_pred = model.predict(X_test)
# print(Y_test_pred)
Y_pred = model.predict(X)
# print(Y_pred)
Y_pred = np.argmax(Y_pred, axis=-1)
print(Y_pred)

first_layer_weights = model.layers[0].get_weights()[0]
print(first_layer_weights)
