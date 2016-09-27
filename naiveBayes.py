from __future__ import division
import collections
import math
import random

class Model:
    def __init__(self, arffFile):
        self.trainingFile = arffFile
        self.features = {}  # all feature names and their possible values (including the class label)
        self.featureNameList = []  # this is to maintain the order of features as in the arff
        self.featureCounts = collections.defaultdict(
            lambda: 1)  # contains tuples of the form (label, feature_name, feature_value)
        self.featureVectors = []  # contains all the values and the label as the last entry
        self.labelCounts = collections.defaultdict(lambda: 0)  # these will be smoothed later
        self.m=1

    def TrainClassifier(self):
        for fv in self.training:
            self.labelCounts[fv[self.classIdx]] += 1  # udpate count of the label
            for counter in range(len(fv)):
                if counter!=self.classIdx:
                    self.featureCounts[(fv[self.classIdx], self.featureNameList[counter], fv[counter])] += 1

    def Classify(self, featureVector):  # featureVector is a simple list like the ones that we use to train
        probabilityPerLabel = {}
        for label in self.labelCounts:
            logProb = 0
            for featureValue in featureVector:
                if self.featureNameList[featureVector.index(featureValue)]!='class':
                    logProb += math.log(
                        (self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]+
                         self.m*1.0/len(self.features[self.featureNameList[featureVector.index(featureValue)]]))
                        /(self.labelCounts[label]+self.m))
            probabilityPerLabel[label] = ((self.labelCounts[label]+self.m*1.0/len(self.labelCounts))/ (sum(self.labelCounts.values())+self.m)) * math.exp(logProb)
        #print probabilityPerLabel
        return max(probabilityPerLabel, key=lambda classLabel: probabilityPerLabel[classLabel])

    def GetValues(self,percentage):
        file = open(self.trainingFile, 'r')
        for line in file:
            if line[0] != '@':  # start of actual data
                self.featureVectors.append(line.strip().lower().split(','))
            else:  # feature definitions
                if line.strip().lower().find('@data') == -1 and (not line.lower().startswith('@relation')):
                    self.featureNameList.append(line.strip().split()[1])
                    self.features[self.featureNameList[len(self.featureNameList) - 1]] = line[
                                                                                         line.find('{') + 1: line.find(
                                                                                             '}')].strip().split(',')

        random.shuffle(self.featureVectors)
        split = int(len(self.featureVectors) * percentage)
        self.training = self.featureVectors[:split]
        self.testing = self.featureVectors[split:]
        self.classIdx = self.featureNameList.index('class')
        file.close()

    def TestClassifier(self):
        count=0
        for vector in self.testing:
            #print "classifier: " + self.Classify(vector) + " given " + vector[self.classIdx]
            if self.Classify(vector)==vector[self.classIdx]:
                count+=1

        print count/len(self.testing)

if __name__ == "__main__":
    model = Model("mushroom.arff")
    model.GetValues(0.8)
    model.TrainClassifier()
    model.TestClassifier()