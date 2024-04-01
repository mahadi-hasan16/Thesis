import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import signal
from biosppy.signals import ecg
folder_path = 'D:\\Python\\Dataset\\Math_Test\\'
df=pd.read_csv(folder_path+'3_ecg_3.csv')
y=df['ecg2'].values

fs=256

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


Ecg=to_mv(y)

plt.subplot(3,1,1)
plt.title("ECG with Raw Data")
plt.plot(Ecg)
plt.grid(True)

noiseFree= notchFilter(50.0,Ecg)
# print(noiseFree)

plt.subplot(3,1,2)
plt.title("ECG with noise free Data")
plt.plot(noiseFree)
plt.grid(True)

pureEcg=butter(0.05,35.0,noiseFree)

# print(type(ecg))

results = ecg.ecg(signal=pureEcg, sampling_rate=fs, show=False)
rpeaks = results["rpeaks"]
rpeaks=np.array(rpeaks)  # Get R peak indices
print(len(y))

plt.subplot(3,1,3)
plt.title("Pure ECG")
plt.plot(pureEcg)
plt.grid(True)


# plt.show()