## Usage
The parsing scripts in this folder can be used to select the data related to specific devices and create other datasets with the selected devices.
In particolar:
* __parsing_2d.py__: specifying the ids of two devices as command line argument, a dataset with only the data related to them is created
* __parsing_3d.py__: specifying the ids of three devices as command line argument, a dataset with only the data related to them is created
* __parsing_4d.py__: specifying the ids of four devices as command line argument, a dataset with only the data related to them is created
* __parsing_5d.py__: specifying the ids of five devices as command line argument, a dataset with only the data related to them is created
* __parsing_10d.py__: specifying the ids of ten devices as command line argument, a dataset with only the data related to them is created
* __parsing_group_2.py__: the script takes three paramenters as command line argument. The first one is the number of devices per group, the second one is the first device_id of the first group, while the third one is the first device_id of the second group. The script produces a dataset with only two ids, all the elements of the first group takes the id of the first one, while all the elements of the second group takes the id of the second one, enabling simple binary classification among groups.




Note that the path for complete dataset and saving folder should be changed properly.
If the dataset changes, for example if one or more features are added, all the scripts should be properly adapted.