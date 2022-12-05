# Fully-automatic-AF-screening
A demo for testing deep neural network for automatically screen atrial fibrillation (AF) patients using RR intervals. Companion code to the paper "Fully automatic screening of patients with atrial fibrillation in long-term heartbeat monitoring using deep learning".

## Requirement
This code was tested on Python 3.6 with Tensorflow-gpu 2.0.0. In addition, numpy 1.19.3,pandas 1.1.1, biosppy 0.7.3 and wfdb 3.3.0 were also used. 

## Files
The folder Demo/data contains the public testing data that were used in the paper and the folder Demo/code contains all the corresponding scripts for evaluating the performance of the model. In the folder Demo/code, load_data.py is used for extracting the RR-interval sequence from the testing data, test.py contains the codes for testing input samples with the trained model, the model structure of this paper is provided in model.py and model.h5 is the trained model. The file mian.py is the script for executing all the processes and the testing results including sensitivity, specificity and accuracy at both "sample-level" and "patient level" will be saved in Demo/results/output.txt.

## Model
TThe model used in the paper is a convolutional, long short-term memory, and fully connected deep neural network (CLDNN), the architecture of the model is shown in Figure 1. The model receives an input tensor with dimension (N, 90, 1), and returns an output tensor with dimension (N, 2), for which N is the batch size. The model presented in Demo/code/model.h5 is a trained model and can be directly used to test the data.

![image](https://github.com/hustzp/Fully-automatic-AF-screening/blob/main/Source/Figure%201.png?raw=true)

Figure 1. Architecture of the CLDNN model.

Input of the model: shape = (N, 90, 1). The input tensor should contain the 90 points of the RR interval sample. 90 RR interval samples were extracted from the test data. All RR intervals are represented at the scale 1 ms, therefore, if the input data are in s it should be multiplied by 1000 before feeding it to the neural network model.
Output of the model: shape = (N, 2). The output contains two probabilities of AF and not AF, between 0 and 1, and sum to 1. 

## Test data
/data contains testing data of three public datasets that are used in this paper, including AFDB, NSRDB, NSRRRIDB. The python package wfdb can be used to read and process the datasets and obtain the data of ECG signals. The files in AFDB and NSRDB are larger than 25 MB and cannot be uploaded to this website. `For convenience, we also uploaded all the three datasets and the code to the following website.` 
https://drive.google.com/file/d/18B32eUOyIWOifUr1dqyI6tg_AeonKggF/view?usp=share_link

`It is worth noting that the data of the three datasets must be complete before running this demo.`

## Results
The results of AF detection in “sample-level” and “patient level” are stored in the folder Demo/results/output.txt.

## Installation guide for running Demo
1, Install Python 3.6 with Tensorflow-gpu 2.0.0. Then, install the following libraries:  numpy 1.19.3, pandas 1.1.1, biosppy 0.7.3, and wfdb 3.3.0.  
2, Download the Demo.zip file and extract the files from it. Using the command line:  
	$ unzip Demo.zip  
3, Run the script of main.py, using the command line:  
	$ python main.py  
After running, there will be a output.txt file in the folder Demo/results, as shown in Figure 2.

![image](https://github.com/hustzp/AF-detection/blob/main/source/Figure%202.png?raw=true)  
Figure 2. The output of the Demo.

## License
The publicly available datasets MIT-BIH AF database, MIT-BIH NSR database, and NSR RR Interval database are available at：
https://physionet.org/content/afdb/1.0.0/, https://physionet.org/content/nsrdb/1.0.0/, and https://physionet.org/content/nsr2db/1.0.0/, respectively. The use of these data must comply with the provisions of these public data sets. This code is to be used only for educational and research purposes. Any commercial use, including the distribution, sale, lease, license, or other transfer of the code to a third party, is prohibited. 
