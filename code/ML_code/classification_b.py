import os
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.manifold import TSNE
# import seaborn as sns
import matplotlib.pyplot as plt

#=====================================================================
# def tSNEPlots(strOutputFile, df):
#     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#     tsne_results = tsne.fit_transform(df)

#     df['Feature-one'] = tsne_results[:,0]
#     df['Feature-two'] = tsne_results[:,1]

#     plt.figure(figsize=(5, 4))
#     sns.scatterplot(
#         x="Feature-one", y="Feature-two",
#         hue="Labels",
#         #palette=sns.color_palette("hls", 10),
#         data=df,
#         legend="full",
#         #alpha=0.3
#     )
#     plt.legend(loc='upper right')
#     plt.grid(linestyle='dotted')
#     plt.savefig(strOutputFile, dpi=300, bbox_inches='tight')
#     plt.show()

# def TestTSNE():
#     fTestSize = 0.98
#     data_train, data_test, label_train, label_test = train_test_split(
#             dfData.to_numpy(), dfLabel.to_numpy().ravel(), 
#             test_size=fTestSize,  random_state = 42)
#     print('Training:', data_train.shape)


#     listDS = np.hstack((data_train, label_train.reshape(-1, 1)))
#     dfCols = ['Bin_'+str(i) for i in range(data_train.shape[1])]
#     dfCols.append('Labels')
#     #print(dfCols)
#     df = pd.DataFrame(listDS, columns=dfCols)

#     strOutputFile = './tsneData_train.pdf'
#     tSNEPlots(strOutputFile, data_train)

################################################################
class CMultiModel:
    def __init__(self):
        print('CMultiModel::CTOR')

    def SelectClassifier(self, strToken = "RF"):
        if strToken == "DT":
            self.mModel = DecisionTreeClassifier(max_depth=2)
        elif strToken == "SVC":
            self.mModel = SVC(kernel='linear', C=1)
        elif strToken == "NB":
            self.mModel = GaussianNB()
        elif strToken == "RF":
            self.mModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        elif strToken == "MLP":
            self.mModel = MLPClassifier(max_iter=5000,solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1)


    def Run(self, arrDS, arrLabels, fTestSz, bVerbose = True):
        # X           Y          x_true           y_true
        DataTrain, DataTest, LabelTrain_true, LabelTest_true = train_test_split(
            arrDS, arrLabels.ravel(),
            stratify=arrLabels, test_size=fTestSz, random_state=42)
        
        # print('Training Data: ' + str(DataTrain.shape))
        # print('Testing Data: ' + str(DataTest.shape))
        file.write('Training Data: ' + str(DataTrain.shape)+"\n")
        file.write('Testing Data: ' + str(DataTest.shape)+"\n")

        self.mModel.fit(DataTrain, LabelTrain_true)
        LabelTestPredicted = self.mModel.predict(DataTest)

        tn, fp, fn, tp = confusion_matrix(LabelTest_true, LabelTestPredicted).ravel()

        fPrecision = round(100*tp/(tp+fp), 2)
        fRecall = round(100*tp/(tp+fn), 2)

        # F1_score = f1_score(y_true, y_pred, average='weighted')
        F1_score = 2 *(fPrecision *fRecall) / (fPrecision + fRecall)
        F1_score = round(100 * F1_score, 2)
        
        # model accuracy for DataTest - 
        # Return the mean accuracy on the given test data and labels.
        fAccuracy =  self.mModel.score(DataTest, LabelTest_true)
        fAccuracy = round(100 * fAccuracy, 2)

        txt='Model:'+ str(self.mModel)+ '\nA:'+ str(fAccuracy)+' P:'+str(fPrecision)+ ' R:' + str(fRecall)+' F1:'+ str(F1_score)+"\n"
        file.write(txt)


        display = PrecisionRecallDisplay.from_estimator(self.mModel, DataTest, LabelTest_true, name=self.mModel)
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        plt.grid(True)
        plt.savefig(sys.argv[1][:len(sys.argv[1])-4]+str(self.mModel)[:4]+"_"+str("_prg.png")) # path to save the precision-recall graph

################################################################
if __name__ == '__main__':

    file=open("log_train_2d_dist.txt", 'a') # path to save the log
    type="---------------------------"+sys.argv[1]+"-------------------------\n"
    print(type)
    file.write(type)

    strFileName=sys.argv[1]
    df = pd.read_csv(strFileName, delimiter=',')
    # print(df.head)

    dfData = df.loc[:,['rssi', 'lqi']]
    dfLabel = df.loc[:,['device_id']]

    print(dfData.shape, dfLabel.shape)
    print(df.groupby('device_id').size())

    # print("-----------SVC-----------")
    # file.write("-----------SVC-----------\n")
    # objCM = CMultiModel()
    # objCM.SelectClassifier("SVC")
    fTestSz = 0.1 # percentage of dataset to use for testing
    # objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    print("-----------DT------------")
    file.write("-----------DT-----------\n")
    objCM = CMultiModel()
    objCM.SelectClassifier("DT")
    # fTestSz = 0.98
    objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    print("-----------NB------------")
    file.write("-----------NB-----------\n")
    objCM = CMultiModel()
    objCM.SelectClassifier("NB")
    # fTestSz = 0.98
    objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    print("-----------RF------------")
    file.write("-----------RF-----------\n")
    objCM = CMultiModel()
    objCM.SelectClassifier("RF")
    # fTestSz = 0.98
    objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    print("-----------MLP------------")
    file.write("-----------MLP-----------\n")
    objCM = CMultiModel()
    objCM.SelectClassifier("MLP")
    # fTestSz = 0.98
    objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

