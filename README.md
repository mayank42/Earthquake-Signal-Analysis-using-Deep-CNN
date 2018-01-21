# Earthquake-Signal-Analysis-using-Deep-CNN
This project explores the possibility of earthquake magnitude prediction based on foreshocks using Deep Convolutional Neural Networks.

### Data Description

The following image is a typical earthquake signal and illustrates the data we are gonna deal with.
![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Data%20Description/vt_signal.png)

Earthquake signals are recorded as three directional waves, with directions namely ( vertical, east-west, north-south ). The above is a vertical signal. The signals can be interpreted as coming in three different channels ( vt,es, ns ). These are fed into the CNN as regular images are fed with three distinct channels.

### Data Augmentation

Since earthquakes are quite rare ( thank god ), the dataset for earthquake signals is small. After considerable search for various methodoligies for augmenting signal data, I have found that the technique of timestretching signals is the most popular. The idea is simple - we stretch the signal to create a new one. But simple resampling would distort the frequencies present in the signal. Therefore we need to stretch the signal without affecting its inherent frequencies. This is done using the phase vocoder technique, more details about which can be found here: https://en.wikipedia.org/wiki/Phase_vocoder

A typical signal stretch would change the signal as shown in the figure below:
![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Data%20Description/stretched_signals.png)

The signals have been stretched with 100 factors in the range [0.8,1.5].

### Filtering for foreshocks:

The foreshock part of a signal is technically defined as the part before the onset of P-wave. The P-wave is usually detected using sta-lta algorithm which basically detects differences between long and short term averages of the signal. The algorithm requires the size of short and long average windows and needs to be adjusted to the sensitivity we want. I have tuned all the parameters with respect to the data and can be seen directly from the code. A typical foreshock point is shown in the figure below:

![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Data%20Description/sta_lta.png)

The part before the red line is what we'll train our network on.

### Learning pipeline

Finally our net design is more or less a tweak of AlexNet. The design is as written in the uploaded jupyter notebook ( Dummy Pipeline ). Since the training on a jupyter notebook on my laptop wasn't feasible, I've trained the net on our departmental GPU server with the script train.py. The jupyter notebook has been uploaded here just to show how we have proceeded and does not represent the complete training process. The learning pipeline is very clear from the notebook. The same has been rigorously coded in the script train.py. Due to large data filtering process and high time taking ( 2 days approx. ) training process, the two have been done seperately and you can see the data has been loaded from a pickle file in train.py. The final keras model has been saved as hd5 file ( Results/EqModel.hd5 ) which also has been uploaded.

### Results

The data had been divided into ten classes, namely the magnitudes from 0-9. The integer part of the orignal signal magnitude has been taken as the class for that signal. Since hard pin-point accuracy in such a case would be not be a good measure of model accuracy, I have plotted a accuracy diagram where the model prediction has been compared with the nearby integer values of the orignal signal magnitude. How much nearby is denoted the soft margin in the diagram. For example, if the orignal magnitude is 5.9 and given a margin of 0.5, soft accuracy will be defined as the prediction been equal to int(5.4) or int(5.9) or int(6.4). This seems reasonable as long as our margin is not very high. I have plotted a accuracy vs maring diagram which clearly suggests the appropriateness of the model.
![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Results/accuracy_plot.png)

Also the training testing loss curve is shown below. Since the signals were of different length, I could not have just fed the data into the keras api. So the training has been done on an incremental basis, fit signals one by one, and testing been done after every 10 signals fitted. Also, in the figure I've just shown the curve on the first 250 iterations of training although the number of iterations run through approx 20*31000 times. The part in the figure depicts the portion of substantial drop.

![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Results/loss_curve.png)

I have also plotted the frequency response of some of the first layer filters which can give a little more insight to the data itself.
![alt text](https://github.com/mayank42/Earthquake-Signal-Analysis-using-Deep-CNN/blob/master/Results/FilterMap.png)
The red curve is for vt channel, blue for ns channel and green for ew channel.

