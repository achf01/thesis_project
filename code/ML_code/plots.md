## Usage

### Plots
The code described can be used to plot different information in order to have a visual representation of them.
* __dataset_plot.py__ is needed to compute a scatterplot of the dataset. It takes the dataset in .csv format as first command line argument and then plots it based on the rssi and lqi value. The point of the same class are represented with the same color. There is no constraint on the number of classes present in the dataset.
* __classifier_comp_plot.py__ is needed to represent the dataset and how different ML algorithms perfoms on it. It considers only datasets with two classes, since the algorithms are set up to compute binary classification. It can be used as a suggestion to understand which algorithms to avoid. It takes two command line arguments that will be considered datsets in .csv format.
The code can be modified in order to display a different number of datasets and also to use a different set of algorithms.

### Classification algorithms
The code described can be used to run different ML model with training and testing. The performance results are printed on a log file and the precision recall graph is saved. The paths for saving can be easily changed. 
* __classification_b.py__ is used for binary classification. It takes as first command line argument the path of the binary dataset in .csv format
* __classification_nb.py__ is used for multi-class classification. It takes as first command line argument the path of the dataset in .csv format and as second argument an abbreviation of the model to use for training
