import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
import scipy.signal as signal
import time
import serial
#This script is used for capturing live data to be used to train the UBM

def save_audio_waveform(waveform, sampling_rate, filename):
    sf.write(filename, waveform, sampling_rate)
    print(f"Audio saved as {filename}")


def play_audio(waveform, sampling_rate):
    sd.play(waveform, sampling_rate)
    sd.wait()
    print("Playback finished.")


def get_smoothed_data(record_seconds=5):
    SAMPLING_RATE = 25000
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("Recording...")
    time.sleep(1)
    audio_data = sd.rec(int(record_seconds * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float64')
    sd.wait()
    print("Recording finished.")
    # Voice saved

    audio_data = audio_data.flatten()
    baseline = np.mean(audio_data)
    audio_data -= baseline
    # audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)
    #save_audio_waveform(audio_data, SAMPLING_RATE, filename="Unfiltered2.wav")
    play_audio(audio_data, SAMPLING_RATE)

    return audio_data, SAMPLING_RATE


def capture_audio(duration, sampling_rate):
    smoothed_audio_data, sampling_rate = get_smoothed_data(record_seconds=duration)
    return smoothed_audio_data, sampling_rate


def plot_waveform(waveform, sampling_rate):
    print(waveform)
    if len(waveform) == 0:
        print("No audio data to plot.")
        return
    plt.figure(figsize=(12, 6))
    time_axis = np.arange(0, len(waveform)) / sampling_rate
    plt.plot(time_axis, waveform)
    plt.title('Time Domain Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.show()


def plot_mfccs(mfccs, sampling_rate):
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time')
    plt.ylabel('MFCC Coefficient Index')
    plt.colorbar(format='%+2.0f dB')
    plt.title('MFCCs')
    plt.tight_layout()
    plt.show()

def print_mfccs(mfccs):
    print("MFCCs Array:", mfccs.shape)
    print(mfccs)

def train_gmm_for_person(phrases, duration=5, sampling_rate=44100):
    all_mfccs = []

    # Record phrase
    print(f"Say phrase: '{phrases}'")
    waveform, sample_rate = capture_audio(duration, sampling_rate)
    plot_waveform(waveform, sample_rate)

    redo = input("Retake? Y/N")
    while redo == 'Y':
        print(f"Say phrase: '{phrases}'")
        waveform, sample_rate = capture_audio(duration, sampling_rate)
        plot_waveform(waveform, sample_rate)
        redo = input("Retake? Y/N")

    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    all_mfccs.append(mfccs)
    play_audio(waveform, sample_rate)
    combined_mfccs = np.hstack(all_mfccs)


def main():
    # Arrays for the GMM and Scalers
    gmms = []
    scalers = []
    phrases="Say whatever you want"
    train_gmm_for_person(phrases)

if __name__ == "__main__":
    main()
