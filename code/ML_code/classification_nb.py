import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


class CMultiModel:
    def __init__(self):
        print('CMultiModel::CTOR')

    def SelectClassifier(self, strToken = "RF"):
        if strToken == "DT":
            self.mModel = DecisionTreeClassifier(max_depth=2)
            self.viz=PrecisionRecallCurve(
                self.mModel,
                per_class=True,
                cmap="Set1",
                classes=num
            )
        elif strToken == "SVC":
            self.mModel = SVC(kernel='linear', C=1)
            self.viz=PrecisionRecallCurve(
                self.mModel,
                per_class=True,
                cmap="Set1",
                classes=num
            )
        elif strToken == "NB":
            self.mModel = GaussianNB()
            self.viz=PrecisionRecallCurve(
                self.mModel,
                per_class=True,
                cmap="Set1",
                classes=num
            )
        elif strToken == "RF":
            self.mModel = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
            self.viz=PrecisionRecallCurve(
                self.mModel,
                per_class=True,
                cmap="Set1",
                classes=num
            )
        elif strToken == "MLP":
            self.mModel = MLPClassifier(max_iter=5000,solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1)
            self.viz=PrecisionRecallCurve(
                self.mModel,
                per_class=True,
                cmap="Set1",
                classes=num
            )

    def Run(self, arrDS, arrLabels, fTestSz, bVerbose = True):
        # X           Y          x_true           y_true
        DataTrain, DataTest, LabelTrain_true, LabelTest_true = train_test_split(
            arrDS, arrLabels.ravel(),
            stratify=arrLabels, test_size=fTestSz, random_state=42)
        
        file.write('Training Data: ' + str(DataTrain.shape)+"\n")
        file.write('Testing Data: ' + str(DataTest.shape)+"\n")

        self.mModel.fit(DataTrain, LabelTrain_true)
        LabelTestPredicted = self.mModel.predict(DataTest)

        accuracy = accuracy_score(LabelTest_true, LabelTestPredicted)
        precision = precision_score(LabelTest_true, LabelTestPredicted, average="weighted")
        recall = recall_score(LabelTest_true, LabelTestPredicted, average="weighted")
        f1 = f1_score(LabelTest_true, LabelTestPredicted, average="weighted")
        
        res="A:"+str(accuracy)+" P:"+ str(precision)+" R:"+str(recall)+" f1:"+str(f1)+"\n"
        file.write(res)

        self.viz.fit(DataTrain,LabelTrain_true)
        self.viz.score(DataTest, LabelTest_true)
        self.viz.show(outpath=sys.argv[1][:len(sys.argv[1])-4]+"_"+str(self.mModel)+".png") # path to save the image

################################################################
if __name__ == '__main__':

    file=open("path_log_to_save", 'a')  # path where to save the log
    type="---------------------------"+sys.argv[1]+"-------------------------\n"
    print(type)
    file.write(type)

    
    name=sys.argv[1].split("/")
    name=name[len(name)-1]
    # print(name)
    num=[]
    tmp=name.split("_")
    for el in tmp:
        if("data" in el):
            num.append(str(el[4:]))
        elif(".csv" in el):
            num.append(str(el[:len(el)-4]))
        else:
            num.append(str(el))

    print(num)

    mod=sys.argv[2]

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


    if(sys.argv[2]=="DT"):
        print("-----------DT------------")
        file.write("-----------DT-----------\n")
        objCM = CMultiModel()
        objCM.SelectClassifier("DT")
        # fTestSz = 0.98
        objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    elif(sys.argv[2]=="NB"):
        print("-----------NB------------")
        file.write("-----------NB-----------\n")
        objCM = CMultiModel()
        objCM.SelectClassifier("NB")
        # fTestSz = 0.98
        objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    elif(sys.argv[2]=="RF"):
        print("-----------RF------------")
        file.write("-----------RF-----------\n")
        objCM = CMultiModel()
        objCM.SelectClassifier("RF")
        # fTestSz = 0.98
        objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

    elif(sys.argv[2]=="MLP"):
        print("-----------MLP------------")
        file.write("-----------MLP-----------\n")
        objCM = CMultiModel()
        objCM.SelectClassifier("MLP")
        # fTestSz = 0.98
        objCM.Run(dfData.to_numpy(), dfLabel.to_numpy(), fTestSz)

