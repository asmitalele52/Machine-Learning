import numpy as np
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

def CreateParamTunedModelMap():
    models = {}
    #tuned clf for decision tree
    dtc = DecisionTreeClassifier()
    dtcParameters = [{'max_depth': (8,16,32,64), 'min_samples_split':np.arange(2,12),
                  'min_impurity_decrease':(0.001,0.01,0.1,0.0001), 'max_features':(8,16,32,64)}]
    dtClf = GridSearchCV(dtc, dtcParameters)
    models["Decision Tree Classifier"] = dtClf
    #tune clf for MLP alias artificial neuralNet 
    mlpcParameters=[{'activation' : ('identity', 'logistic', 'tanh', 'relu'),'alpha':(0.00001,0.001,0.1,1),
                     'max_iter':(1500,1000,2000), 'learning_rate':('constant','invscaling','adaptive')}]
    mlpc = MLPClassifier()
    mlpClf = GridSearchCV(mlpc, mlpcParameters)
    models["Neural Net Classifier"] = mlpClf
    #tune clf for svm
    svcParameters = [{'kernel': ['linear', 'rbf', 'poly'], 'C': [1, 10], 'degree':np.arange(1,5), 'gamma':[0.001, 0.0001],
                   'random_state':[0,1], 'max_iter':(1500,1000,2000)}]
    svc = svm.SVC(probability=True)
    clfSvc = GridSearchCV(svc, svcParameters)
    models["SVM Classifier"] = clfSvc
    #tuned clf for Gausian Naive Bayes
    gnbcParameters = {}
    gnbc = GaussianNB()
    gnbcClf = GridSearchCV(gnbc,gnbcParameters)
    models["Gaussian Naive Base Classifier"] = gnbcClf
    #tuned clf for Logistic Regression
    lrcParameters = [{'penalty' : ('l1','l2'),'tol' : (0.03,0.04,0.05), 'max_iter' : (1000, 1500, 2000),
                      'fit_intercept' : (True,False)}]
    #lrc = OneVsRestClassifier(LogisticRegression())
    lrc = LogisticRegression()
    #print(lrc.estimator.get_params().keys())
    lrcClf = GridSearchCV(lrc,lrcParameters)
    models["Logistic Regression Classifier"] = lrcClf
    #tuned clf for KNNClassifier
    knncParameters = [{'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),'p':np.arange(1,5),
                   'n_neighbors':(1,2,3,4,5)}]
    knnc = KNeighborsClassifier()
    knncClf = GridSearchCV(knnc, knncParameters)
    models["KNN Classifier"] = knncClf
    return models
def printAccuracyMeasure(classifier,x_test,y_test):
    score = classifier.score(x_test,y_test)
    print(" Accuracy Measure : " + str(score))
def printConfusionMatrix(classifier,x_test,y_test):
    predictions = classifier.predict(x_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    print("confusion matrix : ")
    print(cm)
def printClassificationReport(classifier,x_test,y_test):
    predictions = classifier.predict(x_test)
    print("classification report : " )
    print(metrics.classification_report(y_test, predictions))
def printBestParameters(classifier):
    print("best parameters : ", classifier.best_params_)
def evaluateModels(models, data):
    for clfName,clf in models.items() :
        print(str(clfName) + ":")
        for i in range(5):
            x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=(i+1)*10)
            clf.fit(x_train,y_train)
            print("iter num : " + str(i+1) + " of data split")
            printBestParameters(clf)
            printConfusionMatrix(clf,x_test,y_test)
            printAccuracyMeasure(clf,x_test,y_test)
            printClassificationReport(clf,x_test,y_test)
            printAreaAndPlotRocCurve(clf,x_test,y_test,i)
def printAreaAndPlotRocCurve(classifier, x_test, y_test, itr):
    probas_ = classifier.predict_proba(x_test)
    colors = ['blue','black','green','yellow','aqua','grey','skyblue','peru','darkorange','saddlebrown']
    for i in range(10) :
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, i],pos_label = i)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, alpha=1,
             label='ROC for class %d (AUC = %0.5f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='red',
             label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    print("\n")
if __name__ == "__main__" :
    data = datasets.load_digits()
    models = CreateParamTunedModelMap()
    evaluateModels(models,data)