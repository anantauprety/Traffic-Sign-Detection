from abc import ABCMeta
import abc
import cv2
import numpy as np
import os


class Classifier:
    """Abstract base class for all classifiers"""
    __metaclass__ = ABCMeta

    @abc.abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abc.abstractmethod
    def evaluate(self, X_test, y_test, visualize=False):
        pass

class MultiClassSVM(Classifier):
    """Multi-class classification using Support Vector Machines
    (SVMs) """
    def __init__(self, num_classes, pre_calc_classifiers=[], mode="one-vs-all", params=None):
         self.num_classes = num_classes
         self.mode = mode
         self.params = params or dict()
         # initialize correct number of classifiers
         self.classifiers = pre_calc_classifiers
         if mode == "one-vs-one":
             # k classes: need k*(k-1)/2 classifiers
             for i in range(self.num_classes*(self.num_classes-1)/2):
                self.classifiers.append(self._getSVM())
         elif mode == "one-vs-all":
             # k classes: need k classifiers
            for i in range(self.num_classes):
                self.classifiers.append(self._getSVM())
         else:
             print("Unknown mode ",mode)


    def _getSVM(self):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_LINEAR)
        # svm.setDegree(0.0)
        # svm.setGamma(0.0)
        # svm.setCoef0(0.0)
        # svm.setC(0)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)
        svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
        return svm

    def fit(self, X_train, y_train, params=None):
        """ fit model to data """
        if params is None:
            params = self.params

        if self.mode == "one-vs-one":
            svm_id=0
            for c1 in range(self.num_classes):
                for c2 in range(c1+1,self.num_classes):
                    y_train_c1 = np.where(y_train==c1)[0]
                    y_train_c2 = np.where(y_train==c2)[0]
                    data_id = np.sort(np.concatenate((y_train_c1, y_train_c2), axis=0))
                    X_train_id = X_train[data_id,:]
                    y_train_id = y_train[data_id]
                    y_train_bin = np.where(y_train_id==c1, 1, 0).flatten()

                    self.classifiers[svm_id].train(X_train_id, y_train_bin)
                    svm_id += 1
        elif self.mode == "one-vs-all":
            for c in range(self.num_classes):
                y_train_bin = np.where(y_train==c,1,0).flatten()
                self.classifiers[c].train(X_train, cv2.ml.ROW_SAMPLE, y_train_bin)

        # params.term_crit = (cv2.TERM_CRITERIA_EPS +  cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    def predict(self, X_test):
        Y_vote = np.zeros((X_test.shape[0], self.num_classes))
        # print('Y_vote',Y_vote.flatten(),sep=',')
        # print('Y_vote shape',Y_vote.shape,sep=',')
        if self.mode == "one-vs-one":
            svm_id = 0
        elif self.mode == "one-vs-all":
            for c in range(self.num_classes):
                 # predict labels
                # print('start')
                confidence, y_hat = self.classifiers[c].predict(X_test)
                # print('conf', confidence)
                # print('y_hat', y_hat)
                y_hat = y_hat.flatten()
                # print('flat hat', y_hat)
                # we vote for c where y_hat is 1
                if np.any(y_hat):
                    Y_vote[np.where(y_hat==1)[0], c] += 1

        return np.argmax(Y_vote, axis=1)


    def evaluate(self, X_test, y_test, visualize=False):
        """"Evaluates model performance"""
        Y_vote = np.zeros((X_test.shape[0], self.num_classes))
        # print('Y_vote',Y_vote.flatten(),sep=',')
        # print('Y_vote shape',Y_vote.shape,sep=',')
        if self.mode == "one-vs-one":
            svm_id = 0
            # for c1 in range(self.numClasses):
            #     for c2 in range(c1+ 1, self.num_classes):
            #         data_id = np.where((y_test==c1) + (y_test==c2))[0]
            #         X_test_id = X_test[data_id,:],:],:]
            #         y_test_id = y_test[data_id]
            #         # predict labels
            #         y_hat = self.classifiers[svm_id].predict_all(X_test_id)
            #         # we vote for c1 where y_hat is 1, and for c2 where
            #         # y_hat is 0
            #         # np.where serves as the inner index into the
            #         # data_id array, which in turn serves as index
            #         # into the Y_vote matrix
            #         Y_vote[data_id[np.where(y_hat==1)[0]],c1] += 1
            #         Y_vote[data_id[np.where(y_hat==0)[0]],c2] += 1
            #         svm_id += 1

        elif self.mode == "one-vs-all":
            for c in range(self.num_classes):
                 # predict labels
                # print('start')
                confidence, y_hat = self.classifiers[c].predict(X_test)
                # print('conf', confidence)
                # print('y_hat', y_hat)
                y_hat = y_hat.flatten()
                # print('flat hat', y_hat)
                # we vote for c where y_hat is 1
                if np.any(y_hat):
                    Y_vote[np.where(y_hat==1)[0], c] += 1
                # print('Y_vote after',Y_vote)
            # # find all rows without votes, pick a class at random
            # no_label = np.where(np.sum(Y_vote, axis=1)==0)[0]
            # print('no label', no_label)
            # Y_vote[no_label,np.random.randint(self.num_classes, size = len(no_label))] = 1
            # print(Y_vote)
            print('y_test', y_test)

        acc = self.__accuracy(y_test, Y_vote )
        conf = self.__confusion(y_test, Y_vote)
        print(acc)
        print(conf)
        return np.argmax(Y_vote, axis=1)

    def __accuracy(self, y_test, y_vote):
        """ Calculates the accuracy based on a vector of ground-truth
        labels (y_test) and a 2D voting matrix (y_vote) of size
        (len(y_test),numClasses). """
        y_hat = np.argmax(y_vote, axis=1)
        print('acc y_hat', y_hat)
        # all cases where predicted class was correct
        mask = (y_hat == y_test)
        return np.count_nonzero(mask)*1./len(y_test)

    def fit_and_save(self, X_train, y_train, params=None, path= 'svms'):
        if not os.path.exists(path):
            print('creating path %s' % path)
            os.mkdir(path)

        self.fit(X_train, y_train, params)

        for num in range(len(self.classifiers)):
            classifier = self.classifiers[num]
            svm_name = 'svm%d.dat' % num
            full_path = os.path.join(path, svm_name)
            classifier.save(full_path)

            print('done saving to ', full_path)

    def __confusion(self,y_test, Y_vote):
        y_hat = np.argmax(Y_vote, axis=1)
        conf = np.zeros((self.num_classes, self.num_classes)).astype(np.int32)
        for c_true in range(self.num_classes):
            for c_pred in range(self.num_classes):
                y_this = np.where((y_test==c_true) * (y_hat==c_pred))
                conf[c_pred,c_true] = np.count_nonzero(y_this)
        return conf


def reload_classifier(path='svms'):

    if os.path.exists(path):
        svm_names = sorted([labelPath for labelPath in os.listdir(path)])
        svmPaths = [os.path.join(path, name) for name in svm_names]
        print(svmPaths)
        loaded_svms = [cv2.ml.SVM_load(svmPath) for svmPath in svmPaths]

        if len(loaded_svms) > 1:
           return MultiClassSVM(len(loaded_svms), loaded_svms)
    print('no svm exists in %s' % path)
    return None





