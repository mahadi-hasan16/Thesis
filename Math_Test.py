import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal
from biosppy.signals import ecg
import os

fs=256

MeanNN=[]
SDNN=[] 
RMSSD=[] 
SDSD=[]
NN50=[] 
pNN50=[]


def to_ms(data):
	return (data*1000)/256

def to_mv(raw_data):
	adc_sensitivity=2420/(2**23-1)
	voltage_mV = (raw_data * adc_sensitivity) / 4
	return voltage_mV


def notchFilter(cutoff,Ecg):
	quality_factor=40.0
	b,a=signal.iirnotch(cutoff,quality_factor,fs)
	noiseFree=signal.filtfilt(b,a,Ecg)
	return noiseFree


def butter(low,high,Ecg):
	Nq=0.5*fs
	lowcut=low/Nq
	highcut=high/Nq
	b,a=signal.butter(5,[lowcut,highcut],btype='band')
	pure_ecg=signal.filtfilt(b,a,Ecg)
	return pure_ecg


# Specify the folder path where your 60 CSV files are located
folder_path = 'D:\\Python\\Dataset\\Math_Test'
# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
	if file_name.endswith('.csv'):
		file_path = os.path.join(folder_path, file_name)
		df = pd.read_csv(file_path)
		y=df['ecg2'].values
		
		try:
			Ecg=to_mv(y)
			noiseFree= notchFilter(50.0,Ecg)
			pureEcg=butter(0.5,40.0,noiseFree)
			results = ecg.ecg(signal=pureEcg, sampling_rate=fs, show=False)
			rpeaks = results["rpeaks"]
			rpeaks=np.array(rpeaks)
			rpeaks=to_ms(rpeaks)
			rr_intervals=np.diff(rpeaks)

			MeanNN.append(np.mean(rr_intervals))
			SDNN.append(np.std(rr_intervals))
			RMSSD.append(np.sqrt(np.mean((rr_intervals[1:] - rr_intervals[:-1])**2)))
			SDSD.append(np.mean(np.abs(rr_intervals[2:] - rr_intervals[:-2])))
			NN50.append(np.sum(rr_intervals > 50.00))
			pnn=(np.sum(rr_intervals > 50.00) / (len(rr_intervals))) * 100
			pNN50.append(pnn)
		except:
			MeanNN.append(0)
			SDNN.append(0)
			RMSSD.append(0)
			SDSD.append(0)
			NN50.append(0)
			pNN50.append(0)
		
		

writeFile='dataset.csv'
data = pd.read_csv(writeFile)

data['Math_MeanNN']=MeanNN
data['Math_SDNN']=SDNN
data['Math_RMSSD']=RMSSD
data['Math_SDSD']=SDNN
data['Math_NN50']=NN50
data['Math_pNN50']=pNN50

data.to_csv(writeFile,index=False)




