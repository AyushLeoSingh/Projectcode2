import numpy
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
import time
#This script is used for checking if pre-recorded data matches pre-trained data

def load_training_data(filename):
    mfccs_transposed = np.loadtxt(filename, delimiter=',')
    print("Training data loaded from", filename)
    return mfccs_transposed


def butter_bandpass_filter(data, lowcut, highcut, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def get_smoothed_data(serial_port='COM13', baud_rate=115200, record_seconds=5,
                      low_cutoff_freq=100, high_cutoff_freq=700):
    SAMPLING_RATE = 2500
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

    N = len(audio_data)
    sampling_rate = 1 / (record_seconds / N)
    if low_cutoff_freq >= 0.5 * sampling_rate or high_cutoff_freq >= 0.5 * sampling_rate:
        raise ValueError("Cutoff frequencies must be less than half the sampling rate.")

    # Filter the audio to reduce noise
    filtered_audio_data = butter_bandpass_filter(audio_data, low_cutoff_freq, high_cutoff_freq, sampling_rate)

    # Smoothen the data using savgol
    # window_length = 51
    # polyorder = 3
    # smoothed_audio_data = savgol_filter(filtered_audio_data, window_length, polyorder)
    # smoothed_audio_data = np.interp(smoothed_audio_data, (smoothed_audio_data.min(), smoothed_audio_data.max()),
    #                                (filtered_audio_data.min(), filtered_audio_data.max()))
    # smoothed_audio_data = smoothed_audio_data[10:len(smoothed_audio_data) - 10]

    return filtered_audio_data, SAMPLING_RATE


def capture_audio(duration=5, sampling_rate=2500):
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


def train_GMM(Name):
    combined_mfccs = load_training_data(Name)
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GaussianMixture(n_components=20, covariance_type='diag')
    # gmm2 = GaussianMixture(n_components=20, covariance_type='diag')
    # gmm2.tol=1e-5
    # print(gmm.tol)
    gmm.fit(mfccs_transposed)
    # gmm2.fit(mfccs_transposed)

    print("GMM Training Complete using loaded data.")
    return gmm, scaler

def fine_train_GMM(Name):
    combined_mfccs = load_training_data(Name)
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GaussianMixture(n_components=20, covariance_type='diag')
    # gmm2 = GaussianMixture(n_components=20, covariance_type='diag')
    gmm.tol=0.001
    # print(gmm.tol)
    gmm.fit(mfccs_transposed)
    # gmm2.fit(mfccs_transposed)

    print("GMM Training Complete using loaded data.")
    return gmm, scaler


def train_gmm_for_person(phrases, duration=5, sampling_rate=2500):
    all_mfccs = []

    # Record phrase
    for phrase in phrases:
        print(f"Say phrase: '{phrase}'")
        waveform, sample_rate = capture_audio(duration, sampling_rate)
        plot_waveform(waveform, sample_rate)

        redo = input("Retake? Y/N")
        while redo == 'Y':
            print(f"Say phrase: '{phrase}'")
            waveform, sample_rate = capture_audio(duration, sampling_rate)
            plot_waveform(waveform, sample_rate)
            redo = input("Retake? Y/N")

        mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        all_mfccs.append(mfccs)

    combined_mfccs = np.hstack(all_mfccs)
    np.savetxt("personX_Data.txt", combined_mfccs, delimiter=',')
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GaussianMixture(n_components=20, covariance_type='diag')
    print(gmm.tol)
    gmm.fit(mfccs_transposed)

    print("GMM Training Complete for this person.")
    return gmm, scaler


def predict_voice(gmms, scalers, duration=5, sampling_rate=2500):
    #waveform, sample_rate = capture_audio(duration, sampling_rate)
    #mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
    mfccs = load_training_data("Ayush_TestingData5.txt")

    best_person = None
    highest_likelihood = -np.inf

    arrlikelyhoods=[]

    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score_samples(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)
    #    print(f"Person {idx + 1} likelihood: {avg_likelihood}")
        arrlikelyhoods.append(avg_likelihood)
    return arrlikelyhoods

    '''
        if avg_likelihood > highest_likelihood:
            highest_likelihood = avg_likelihood
            best_person = idx + 1
    '''

    print(f"Voice is most similar to person {best_person}")



def main():
    # Arrays for the GMM and Scalers
    totals=[]
    for x in range(10):
        gmms = []
        scalers = []
        '''
        gmm, scaler = train_GMM("Rubin_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Ayush2_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Ayush_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Vidic_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Rita_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Pabs_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Kim_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Recal_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
    
        gmm, scaler = train_GMM("Monal_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)
        '''
        gmm, scaler = train_GMM("Ayush_Data3.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = fine_train_GMM("Ayush_Data3.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Mo_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = fine_train_GMM("Mo_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Rubin_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = fine_train_GMM("Rubin_Data.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        totals.append(predict_voice(gmms, scalers))
    totals=np.array(totals)
    column_averages = np.mean(totals, axis=0)
    index_of_max_value = np.argmax(column_averages)

    print("Scores: ", column_averages)
    print("Most similar to person: ", index_of_max_value+1)

    if(numpy.max(column_averages)<-17):
        print("Person not on system")
    else:
        print("Person is on system")


if __name__ == "__main__":
    main()
