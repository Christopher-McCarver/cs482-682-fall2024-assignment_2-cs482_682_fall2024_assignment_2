import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, datasetNum, performTest):
        self.trainingSet = None
        self.testSet = None
        self.modelLogistic = None
        self.modelLinear = None
        self.XTrain = None
        self.yTrain = None
        self.XTest = None
        self.yTest = None
        
        self.performTest = performTest
        self.datasetNum = datasetNum
        self.readCsv(self.datasetNum)

    def readCsv(self, datasetNum):
        if datasetNum == '1':
            trainDatasetFile = 'train_q1_1.csv'
            testDatasetFile = 'test_q1_1.csv'
        elif datasetNum == '2':
            trainDatasetFile = 'train_q1_2.csv'
            testDatasetFile = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.trainingSet = pd.read_csv(trainDatasetFile, sep=',', header=0)
        if self.performTest:
            self.testSet = pd.read_csv(testDatasetFile, sep=',', header=0)
            self.XTest = self.testSet[['exam_score_1', 'exam_score_2']].values
            self.yTest = self.testSet['label'].values

    def modelFitLinear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        self.XTrain = self.trainingSet[['exam_score_1', 'exam_score_2']].values
        self.yTrain = self.trainingSet['label'].values
        self.modelLinear = LinearRegression()
        self.modelLinear.fit(self.XTrain, self.yTrain)

        pass
    
    def modelFitLogistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''

        self.XTrain = self.trainingSet[['exam_score_1', 'exam_score_2']]
        self.yTrain = self.trainingSet['label']
        self.modelLogistic = LogisticRegression()
        self.modelLogistic.fit(self.XTrain, self.yTrain)

        pass
    
    def modelPredictLinear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.modelFitLinear()
        acc = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.modelLinear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.trainingSet is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        if self.XTest is not None:
            # perform prediction here

            yPred = self.modelLinear.predict(self.XTest)
            yPredBinary = (yPred >= 0.5).astype(int) 

            acc = accuracy_score(self.yTest, yPredBinary)
            precision, recall, f1, support = precision_recall_fscore_support(self.yTest, yPredBinary, zero_division=0)

            pass
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [acc, precision, recall, f1, support]

    def modelPredictLogistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.modelFitLogistic()
        acc = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.modelLogistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.trainingSet is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        if self.XTest is not None:
            # perform prediction here
            
            yPred = self.modelLogistic.predict(self.XTest)
            acc = accuracy_score(self.yTest, yPred)
            precision, recall, f1, support = precision_recall_fscore_support(self.yTest, yPred, zero_division=0)

            pass
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [acc, precision, recall, f1, support]

def meshGrid(x, y, h=0.1):
    x_min = x.min() - 1 
    x_max = x.max() + 1
    y_min = y.min() - 1 
    y_max = y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plotDecBound(X, y, model, title, filename):
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm, s=50, label='Data points')

    if len(model.coef_.shape) == 1:
        x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
        y_vals = -(model.coef_[0] * x_vals + model.intercept_) / model.coef_[1]
    else:
        x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
        y_vals = -(model.coef_[0][0] * x_vals + model.intercept_[0]) / model.coef_[0][1]
    
    plt.plot(x_vals, y_vals, 'k--', label='Decision Boundary')
    plt.xlabel('Exam Score 1')
    plt.ylabel('Exam Score 2')
    plt.title(title)
    
    plt.legend()  
    plt.savefig(filename)
    plt.show()



def plotModelDecBound(classif):
    XTrain = classif.trainingSet[['exam_score_1', 'exam_score_2']].values
    yTrain = classif.trainingSet['label'].values
    classif.modelFitLinear()
    plotDecBound(XTrain, yTrain, classif.modelLinear, 'Linear Regression Decision Boundary', 'linear_regression_boundary.png')
    classif.modelFitLogistic()
    plotDecBound(XTrain, yTrain, classif.modelLogistic, 'Logistic Regression Decision Boundary', 'logistic_regression_boundary.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear and Logistic Regression')
    parser.add_argument('-d','--dataset_num', type=str, default="1", choices=["1", "2"], help='Dataset number: 1 or 2')
    parser.add_argument('-t', '--perform_test', action='store_true', help='Perform test if set')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    accLinear = classifier.modelPredictLinear()
    accLogistic = classifier.modelPredictLogistic()
    print("Linear Regression", accLinear)
    print("Logistic Regression", accLogistic)   
    plotModelDecBound(classifier) 
    plt.show()