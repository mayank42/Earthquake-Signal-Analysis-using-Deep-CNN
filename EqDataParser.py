"""This module reads and compiles data in the format required for training
   on the convolution nets.
"""

import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join
from pv import PhaseVocoder
import sys

#CONSTANTS
epsilon = sys.float_info.epsilon

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

class EarthquakeDataParser:
	
	"""This class contains the necessary functions and variables
	   for parsing the earthquake data in the format necessary for 
	   convolution net training.
	"""

	def __init__(self,path,image_data_format,sig_len_thresh, \
				 shortw,longw,sfreq,sta_lta_thresh):
		
		"""
		   Init for this parser. See load_earthquake_data() for param details.
		"""
		self.path = path
		self.image_data_format = image_data_format
		self.sig_len_thresh = sig_len_thresh
		self.shortw = shortw
		self.longw = longw
		self.sfreq = sfreq
		self.sta_lta_thresh = sta_lta_thresh

	def parse_mag(self):

		"""To read data given files.
		   :returns: Tuple of three data lists(vt,ns,ew) of the form [(<mag>,<numpy array of signal>)]

		   All data format locations are specific to data files I have.
		"""
		onlyfiles = [join(self.path,f) for f in listdir(self.path) if isfile(join(self.path, f))]
		vtfiles = [f for f in onlyfiles if f.endswith('.vt')] #Vertical component files			      	
		mag1_data = []
		mag2_data = []
		mag3_data = []
		pi=1
		pf=len(vtfiles)
		for f in vtfiles:
		    temp1=pd.read_csv(f,sep='\n',header=None,encoding='L2')
		    temp2=pd.read_csv(f[:-3]+'.ns',sep='\n',header=None,encoding='L2')
		    temp3=pd.read_csv(f[:-3]+'.ew',sep='\n',header=None,encoding='L2')
		    mag1_data.append(     [float(   temp1.iloc[4][0].split()[-1]            ),temp1.iloc[20:].values.ravel().astype(float)]        )
		    mag2_data.append(     [float(   temp2.iloc[4][0].split()[-1]            ),temp2.iloc[20:].values.ravel().astype(float)]        )
		    mag3_data.append(     [float(   temp3.iloc[4][0].split()[-1]            ),temp3.iloc[20:].values.ravel().astype(float)]        )        
		    sys.stdout.write('Percentage finished: %d %% \r'%int(100.0*pi/pf))
		    sys.stdout.flush()
		    pi=pi+1
		print('\n')
		return mag1_data,mag2_data,mag3_data

	def sta_lta(self,sig):
		"""Implementation of sta_lta algorithm, adapted to the current dataset.
		   
		   :returns: Point of occurence of P-wave ( Adjusted to 90% of detected value )
		   :rtype: int
		   
		   All default values have been adapted for the dataset, I am working on.
		"""
		shortw = self.shortw
		longw = self.longw
		sfreq = self.sfreq
		sta_lta_thresh = self.sta_lta_thresh				
		sw = int(sfreq*shortw)
		lw = int(sfreq*longw)
		ma_sw = np.convolve(np.abs(sig), np.ones((sw,))/sw, mode='valid')
		ma_lw = np.convolve(np.abs(sig), np.ones((lw,))/lw, mode='valid')
		ma_lw[ma_lw == 0] = epsilon
		return np.argmax(np.abs(ma_sw[:len(ma_lw)]/ma_lw)<sta_lta_thresh)
	
	def filter_channelize(self,data):
		
		"""Filters signal data w.r.t required sig_len_thresh ( see load_earthquake_data ). \
		   Also channelizes data into appropriate format for training.
		   :param data: Three-tuple of directional signals.
		   :returns: filtered and channelized data w.r.t. sig_len_thresh and image_data_format
		   :rtype: Two-tuple of form <signals,mag> == <datax,datay>
		"""
		sig_len_thresh = self.sig_len_thresh
		iformat = self.image_data_format
		mag1 = data[0]
		mag2 = data[1]
		mag3 = data[2]
		cdatax = []
		cdatay = []
		if iformat == 'channels_last':
			for i in range(len(mag1)):
				res = self.sta_lta(mag1[i][1])
				if res >= sig_len_thresh:
					tempsig = np.array([[mag1[i][1][0],mag2[i][1][0],mag3[i][1][0]]])
					for j in range(1,res+1):
						tempsig = np.append(tempsig,[[mag1[i][1][j],mag2[i][1][j],mag3[i][1][j]]],axis=0)
					cdatax.append(tempsig)
					cdatay.append(mag1[i][0])
				sys.stdout.write('Percentage finished: %d %% \r'%int(100.0*(i+1)/len(mag1)))
				sys.stdout.flush()
			print('\n')
			cdatax = np.array(cdatax)			
			cdatay = np.array(cdatay)
		else:
			for i in range(len(mag1)):
				res = sta_lta(mag1[i][1])
				if res >= sig_len_thresh:
					tempsig = []
					tempsig.append(np.array([mag1[i][1][j] for j in range(0,res+1)]))
					tempsig.append(np.array([mag2[i][1][j] for j in range(0,res+1)]))
					tempsig.append(np.array([mag3[i][1][j] for j in range(0,res+1)]))
					tempsig = np.array(tempsig)
					cdatax.append(tempsig)
					cdatay.append(mag1[i][0])
				sys.stdout.write('Percentage finished: %d %% \r'%int(100.0*(i+1)/len(mag1)))
				sys.stdout.flush()
			print('\n')
			cdatax = np.array(cdatax)			
			cdatay = np.array(cdatay)
		
		cdatax,cdatay = shuffle_in_unison(cdatax,cdatay)
		return (cdatax,cdatay)					
			

def load_earthquake_data(path,image_data_format,timestrech_factors,sig_len_thresh=500, \
				 		 shortw=0.05,longw=0.3,sfreq=200,sta_lta_thresh=0.07):
	
	"""Parses data into a trainable format.
	   :param path: Path of raw data folder relative to current directory.
	   :param image_data_format: Keras backend image format requirement \
								 for training.
	   :param timestrech_factors: List like object of timestrching factors \
								  for signal augmentation.
	   :param sig_len_thresh: Minimum signal length for training \
							  after trimming at 90% of P-wave onset.
	   :param shortw: Length of short window in seconds.
	   :param longw: Length of long window in seconds.
	   :param sfreq: Sampling frequency of signal in Hz.
	   :param sta_lta_thresh: Sensitivity of algorithm. Lower values \
				           	  will detect changes in signal early.
	   :type image_data_format: string
	   :type timestrech_factors: list-like
	   :type sig_len_thresh: int

	   All default values have been adapted to the working data set.
	"""
	parser = EarthquakeDataParser(path,image_data_format,sig_len_thresh, \
								  shortw,longw,sfreq,sta_lta_thresh)
	alphas = timestrech_factors 
	print('Reading Data.')
	data = parser.parse_mag()

	# These parameters should be power of two for FFT
	N = 2**10 # Number of channels
	M = 2**10 # Size of window

	w = np.hanning(M-1)# Type of Window (Hanning)
	#w = np.hamming(M-1)# Type of Window (Hamming)
	#w = np.hamm(M-1)			# Type of Window (Hann)
	w = np.append(w, [0])# Make window symmetric about (M-1)/2

	# Synthesis hop factor and hop size
	Os = 4 # Synthesis hop factor 
	Rs = int(N / Os)# Synthesis hop size

	mag1 = []
	mag2 = []
	mag3 = []
	print('Timestretching signals.')
	pci=1
	pcf=len(alphas)*len(data[0])	
	for alpha in alphas:
		pv = PhaseVocoder(N, M, Rs, w, alpha)
		for i in range(0,len(data[0])):
			mag1.append([data[0][i][0],pv.timestretch(data[0][i][1],alpha)])
			mag2.append([data[1][i][0],pv.timestretch(data[1][i][1],alpha)])
			mag3.append([data[2][i][0],pv.timestretch(data[2][i][1],alpha)])
			sys.stdout.write ("Percentage finishied: %d %% \r" % int(100.0*pci/pcf))
			sys.stdout.flush()
			pci=pci+1
	print('\n')
	del data
	data = (mag1,mag2,mag3)
	print('Filtering & channelizing.')
	(cdatax,cdatay) = parser.filter_channelize(data)
	lim = int(0.8*len(cdatax))
	return (cdatax[:lim],cdatay[:lim]),(cdatax[lim:],cdatay[lim:])
