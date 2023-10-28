# -*- coding: utf-8 -*-
"""
@author: Diogo Rodrigues 56153 & Jose Murta 55226
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn import linear_model
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.neighbors import KernelDensity

#Load the data
data = np.loadtxt('SatelliteConjunctionDataRegression.csv',delimiter=',',skiprows=1)

Y = data [:,[6]]
X = data [:,0:6]

#Train and test data before standardization
X_train_og, X_test_og, Y_train_og, Y_test_og = train_test_split(X,Y,test_size = .2)

#Standardization process
scaler = StandardScaler()
X_train= scaler.fit_transform(X_train_og)
Y_train = scaler.fit_transform(Y_train_og)
X_test= scaler.fit_transform(X_test_og)
Y_test = scaler.fit_transform(Y_test_og)

#Cross-validation (Regression) for each fold and returns training and validation errors
def calculate_fold (X_train, Y_train, tr_ix, va_ix,lm):
    X_tr = X_train [tr_ix]
    Y_tr = Y_train [tr_ix]
    X_va = X_train [va_ix]
    Y_va = Y_train [va_ix]
     
    lm.fit (X_tr,Y_tr)
    p_tr = lm.predict(X_tr)
    train_err = mean_squared_error(Y_tr, p_tr)
    
    p_va = lm.predict(X_va)
    va_err = mean_squared_error(Y_va, p_va)
    
    return train_err,va_err, p_tr,p_va
    

#Auxiliar variables
validation_errors = []
training_errors = []
plotCross = []
plotPred = []
folds = 5;
idxs_tr = []
idxs_va = []


best_error = 1000000
#For loop to estimate the best degree
for deg in range(1,7):
    #Auxiliar variables to plot True VS Pred
    tr_err = 0
    va_err = 0
    best_fold_tr = []
    best_fold_va = []
    best_tr_ix = []
    best_va_ix = []
    
    #Polynomial tranformation of the features 
    poly = PolynomialFeatures(deg)
    aux = poly.fit_transform(X_train)
    
    lm = linear_model.LinearRegression()
    sfolder = KFold(n_splits=folds)
   
    #Cross-validation using KFold
    best_va_err = 100000
    for tr_ix, va_ix in sfolder.split(aux, Y_train):
        r, v,p_tr,p_va = calculate_fold(aux, Y_train, tr_ix, va_ix,lm)
        tr_err += r
        va_err += v
        if v < best_va_err:
            best_va_err = v
            best_fold_tr = p_tr
            best_fold_va = p_va
            best_tr_ix = tr_ix 
            best_va_ix = va_ix 
    
    #Variables to be ploted
    training_errors.append(tr_err/folds)
    va_err = va_err/folds
    validation_errors.append(va_err)
    plotCross.append(best_fold_va)
    plotPred.append(best_fold_tr)
    idxs_tr.append(best_tr_ix)
    idxs_va.append(best_va_ix)
    
    #Choose the best hypothesis
    if va_err < best_error:
        best_error = va_err
        best_degree = deg
        model = lm

#Test error
poly_test = PolynomialFeatures(best_degree)
test_pred = model.predict(poly_test.fit_transform(X_test))
test_error = mean_absolute_error(Y_test, test_pred)
print("2 - Regression: Collision avoidance in Space")
print("---- Standardized Data ----")
print("Best polynomial degree: ",best_degree)
print("Estimate the true error: ",test_error)

#For Unstandardized test error
poly_test2 = PolynomialFeatures(best_degree)
model.fit(poly_test2.fit_transform(X_train_og), Y_train_og)
test_pred2 = model.predict(poly_test2.fit_transform(X_test_og))
test_error2 = mean_absolute_error(Y_test_og, test_pred2)
print("---- Unstandardized Data ----")
print("Best polynomial degree: ",best_degree)
print("Estimate the true error: ",test_error2)


########Plot the two plots#####

#REGRESS-TR-VAL
degrees = np.arange(1,7,1)
fig, ax = plt.subplots()
training_line = mlines.Line2D([], [], color='blue', marker='s', label='Training')
validation_line = mlines.Line2D([], [], color='red', marker='x', label='Validation')
ax.legend(handles=[training_line, validation_line])
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.plot(degrees, validation_errors, '-xr')
plt.plot(degrees, training_errors, '-sb')
plt.yscale('log')
plt.show()

#REGRESS-PRED-VS-TRUE
counter = 0
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
fig.suptitle('Horizontally stacked subplots')
fig.subplots_adjust(right=1.5, hspace=0.4)
x = np.linspace(min(Y_train),max(Y_train),100)
y = x
for i in range(0,2):
    for j in range(0,3):
        axs[i,j].set_title('Degree ' + str(counter+1))
        axs[i,j].set_xlabel('True')
        axs[i,j].set_ylabel('Predicted')
        train_points = mlines.Line2D([], [], color='blue', marker='.', label='Training', linestyle ='None')
        validation_points = mlines.Line2D([], [], color='red', marker='.', label='Validation', linestyle ='None')
        axs[i,j].legend(handles=[train_points, validation_points])
        
        axs[i,j].plot(Y_train[idxs_va[counter]], plotCross[counter], '.r',  markersize=3)
        axs[i,j].plot(Y_train [idxs_tr[counter]], plotPred[counter], '.b', markersize=3)
        axs[i,j].plot(x,y, '--', color = (0,1,0))
        axs[i,j].axis([min(Y_train)*1.1, max(Y_train)*1.1, 
                       min(min(plotCross[counter]), min(plotPred[counter]))*1.1,
                       max(max(plotCross[counter]), max(plotPred[counter]))*1.1])
        counter +=1

plt.show()


###################################Exercise 3#####################################################
print("-----------------------------------------------------")
print("3 - Classification: Detecting Bank note fraud")

#Load and suffle the training and test data
data_training = np.loadtxt('TP1_train.tsv')
np.random.shuffle(data_training)
data_test = np.loadtxt('TP1_test.tsv')
np.random.shuffle(data_test)

# Training Features and test labels
features_training = data_training[:, 0:4]
features_training = (features_training - np.mean(features_training, 0)) / np.std(features_training, 0)
labels_training = data_training[:, 4]

# Test Features and test labels
features_test = data_test[:, 0:4]
features_test = (features_test - np.mean(features_test, 0)) /np.std(features_test, 0)
labels_test = data_test[:, 4]


###### Functions #####

#Cross-validation (Classifier) for each fold and returns training and validation errors
def calc_fold(h, features_training, labels_training, tr_ix, va_ix):
    features_tr = features_training[tr_ix]
    features_tr_1 = features_tr[labels_training[tr_ix] == 1]
    features_tr_0 = features_tr[labels_training[tr_ix] == 0]
    prob_1 = np.log(len(features_tr_1)/len(features_tr))
    prob_0 = np.log(len(features_tr_0)/len(features_tr))
    kde1 = KernelDensity(bandwidth=h)
    kde0 = KernelDensity(bandwidth=h)
    #Fit by column
    for i in range(len(features_tr[0, :])):
        kde1.fit(features_tr_1[:, [i]])
        kde0.fit(features_tr_0[:, [i]])
        prob_1 += kde1.score_samples(features_training[:, [i]])
        prob_0 += kde0.score_samples(features_training[:, [i]])

    predicted_labels = []
    for i in range(len(prob_1)):
        if (prob_1[i] >= prob_0[i]):
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    training_accuracy = accuracy_score(labels_training[tr_ix], np.array(predicted_labels)[tr_ix])
    validation_accuracy = accuracy_score(labels_training[va_ix], np.array(predicted_labels)[va_ix])
    return 1-training_accuracy, 1-validation_accuracy


# Normal Test
def normal_test(labels_test, predicted_labels):
    N = len(labels_test)
    diff = labels_test - predicted_labels
    misclass = np.count_nonzero((diff == 1) | (diff == -1))
    p_misclass = misclass/N
    sigma = np.sqrt(N*(p_misclass)*(1-(p_misclass)))
    minus = misclass - 1.96*sigma
    plus = misclass + 1.96*sigma
    return minus, plus, misclass, 1.96*sigma

# Mcnemar Test
def mcnemar_test(labels_test, predicted_labels, predicted_labelsGNB):
    e01 = 0
    e10 = 0
    for i in range(len(labels_test)):
        if predicted_labels[i] != labels_test[i] and predicted_labelsGNB[i] == labels_test[i]:
            e01 += 1
        elif predicted_labels[i] == labels_test[i] and predicted_labelsGNB[i] != labels_test[i]:
            e10 += 1
    return ((np.absolute(e01-e10)-1)**2)/(e01+e10)


#Auxiliar arrays for plots of training and validation errors
training_errors = []
validation_errors = []

#Cross validation using Stratified kfold
folds = 5
kf = StratifiedKFold(n_splits=folds)
bandwidths = np.arange(0.02, 0.62, 0.02)
best_error = 1000000
choosen_bandwith = 0
for h in bandwidths:
    aux_h = round(h, 3)
    tr_err = 0
    va_err = 0
    for tr_ix, va_ix in kf.split(features_training, labels_training):
        r, v = calc_fold(aux_h, features_training,
                         labels_training, tr_ix, va_ix)
        tr_err += r
        va_err += v
    training_errors.append(tr_err/folds)
    va_err = va_err/folds
    validation_errors.append(va_err)
    if va_err < best_error:
        best_error = va_err
        choosen_bandwith = aux_h

print("Best bandwidth value:", choosen_bandwith)
# Plot training vs cross validation error, parameter h tuning
fig, ax = plt.subplots()
training_line_h = mlines.Line2D([], [], color='blue', label='Training error')
validation_line_h = mlines.Line2D([], [], color='red',  label='Cross-Validation error')
ax.legend(handles=[training_line_h, validation_line_h])
plt.xlabel('Bandwidth')
plt.ylabel('Error ( 1 - accuracy )')
plt.plot(bandwidths, training_errors, '-b')
plt.plot(bandwidths, validation_errors, '-r')
plt.show()

# Estimate the true error
kde1 = KernelDensity(bandwidth=choosen_bandwith)
kde0 = KernelDensity(bandwidth=choosen_bandwith)
features_tr_1 = features_training[labels_training == 1]
features_tr_0 = features_training[labels_training == 0]
prob_1 = np.log(len(features_tr_1)/len(features_training))
prob_0 = np.log(len(features_tr_0)/len(features_training))
for i in range(len(features_training[0, :])):
    kde1.fit(features_tr_1[:, [i]])
    kde0.fit(features_tr_0[:, [i]])
    prob_1 += kde1.score_samples(features_test[:, [i]])
    prob_0 += kde0.score_samples(features_test[:, [i]])

predicted_labels = []
for i in range(len(prob_1)):
    if (prob_1[i] >= prob_0[i]):
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

test_accuracy = accuracy_score(labels_test, predicted_labels)
true_error = 1 - test_accuracy
print("Estimate the true error:", true_error)

# Gaussian NB SkLearn
gnb = GaussianNB()
gnb.fit(features_training, labels_training)
predictionGNB = gnb.predict(features_test)
accuracyGNB = gnb.score(features_test, labels_test)
errorGNB = 1 - accuracyGNB
print("Error of the gaussianNB: ", errorGNB)

# Normal Test
minusNB, plusNB, misc1, sig1 = normal_test(labels_test, predicted_labels)
minusGNB, plusGNB, misc2, sig2 = normal_test(labels_test, predictionGNB)
#print("Normal Test NB: ", minusNB, plusNB)
print("Normal Test NB: "+str(misc1)+" - "+str(sig1)+" // "+str(misc1)+" + "+str(sig1))
#print("Normal Test GNB: ", minusGNB, plusGNB)
print("Normal Test GNB: "+str(misc2)+" - "+str(sig2)+" // "+str(misc2)+" + "+str(sig2))

#Mccemar Test
mcnemar_t = mcnemar_test(labels_test, predicted_labels, predictionGNB)
print("Mcnemar's test (NB VS GNB): ", mcnemar_t)
