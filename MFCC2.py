import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
import time
#This script is used for capturing training data for the mic

def load_training_data(filename):
    mfccs_transposed = np.loadtxt(filename, delimiter=',')
    print("Training data loaded from", filename)
    return mfccs_transposed


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data)


def get_smoothed_data(serial_port='COM13', baud_rate=115200, record_seconds=5,
                      low_cutoff_freq=100, high_cutoff_freq=700):
    SAMPLING_RATE = 21000
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
    #Voice saved


    audio_data = audio_data.flatten()
    baseline = np.mean(audio_data)
    audio_data -= baseline

    return audio_data, SAMPLING_RATE


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

def train_GMM_wav(wavform):
    pass


def train_GMM(Name):
    combined_mfccs = load_training_data(Name)
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GaussianMixture(n_components=10, covariance_type='diag')
    #gmm2 = GaussianMixture(n_components=20, covariance_type='diag')
    #gmm2.tol=1e-5
    #print(gmm.tol)
    gmm.fit(mfccs_transposed)
    #gmm2.fit(mfccs_transposed)

    print("GMM Training Complete using loaded data.")
    return gmm, scaler


def train_gmm_for_person(phrases, duration=5, sampling_rate=21000):
    all_mfccs = []

    #Record phrase
    for phrase in phrases:
        print(f"Say phrase: '{phrase}'")
        waveform, sample_rate = capture_audio(duration, sampling_rate)
        lowcut = 40
        highcut = 1700
        filtered_audio = bandpass_filter(waveform, lowcut, highcut, sampling_rate)
        waveform = filtered_audio
        #sd.play(waveform, samplerate=sampling_rate)
        #sd.wait()  # Wait until playback is done
        plot_waveform(waveform, sample_rate)

        redo=input("Retake? Y/N")
        while redo=='Y':
            print(f"Say phrase: '{phrase}'")
            waveform, sample_rate = capture_audio(duration, sampling_rate)
            plot_waveform(waveform, sample_rate)
            redo = input("Retake? Y/N")


        mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        all_mfccs.append(mfccs)

    combined_mfccs = np.hstack(all_mfccs)
    np.savetxt("Andre3_usb.txt", combined_mfccs, delimiter=',')

    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GaussianMixture(n_components=20, covariance_type='diag')
    print(gmm.tol)
    gmm.fit(mfccs_transposed)

    print("GMM Training Complete for this person.")
    return gmm, scaler


def predict_voice(gmms, scalers, duration=5, sampling_rate=21000):
    waveform, sample_rate = capture_audio(duration, sampling_rate)
    lowcut = 40
    highcut = 1700
    filtered_audio = bandpass_filter(waveform, lowcut, highcut, sampling_rate)
    waveform = filtered_audio
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

    best_person = None
    highest_likelihood = -np.inf

    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score_samples(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)
        print(f"Person {idx + 1} likelihood: {avg_likelihood}")

        if avg_likelihood > highest_likelihood:
            highest_likelihood = avg_likelihood
            best_person = idx + 1

    print(f"Voice is most similar to person {best_person}")


def main():
    #First enter the number of people to train
    print("Enter number of people to train: ")
    num_people = int(input().strip())

    #These are the phrases to train the system, NB add more
    phrases = [
        "The juice of lemons makes a refreshing fine punch that is globally recognized as lemonade in certain countries worldwide",
        "Carefully glue the large sheet of paper to the dark blue background to create a cool visual effect that is awesome for portraits",
        "The quick brown fox jumps over the lazy, sleeping dog that lies sprawled in the sun on a hot yet dry summers eve",
        "The rough birch canoe smoothly slid over the well-polished planks of the new dock by the black swan lake in New Jersey",
        "White rice is often served in oval yet round bowls made of fibre glass or cherry brown wood of pine nature",
        "The bright yellow sun sets slowly behind the tall, green hills in the evening sky of purple misty clouds and smoke pollution.",
        "A gentle breeze rustles through the colorful autumn leaves as they fall gracefully to the ground while slowly decaying.",
        "She carefully measures the ingredients and mixes them together to create a delicious homemade black forrest chocolate cake.",
        "The train moves steadily along the tracks, passing through picturesque landscapes and quaint villages as the smoke leaves the chimney.",
        "He reads a captivating book under the soft glow of a warm, cozy lamp in the quiet room in the mental asylum at Arkham."
    ]
    phrases2 = [
        "A majestic waterfall cascades down the rugged mountainside, creating a symphony of sounds as the water crashes into the crystal-clear pool below.",
        "The intricate patterns of the stained glass window cast colorful reflections on the marble floor as the sunlight streams through in the early morning.",
        "As the gentle waves lap against the shore, the distant sound of seagulls fills the air, creating a serene atmosphere on the deserted beach at dusk.",
        "The tall, ancient oak tree stands proudly in the middle of the meadow, its thick branches providing shade to the grazing deer on the soft grass below.",
        "In the bustling market square, vendors shout their wares, and the smell of fresh bread and spices mingles with the laughter of children playing nearby.",
        "The old grandfather clock ticks steadily in the corner of the room, marking the passage of time as the fire crackles softly in the hearth beside it.",
        "A soft rain begins to fall, pattering gently on the rooftops and creating small ripples in the puddles that have formed on the cobblestone streets.",
        "The brilliant display of fireworks lights up the night sky, their vibrant colors reflected in the wide river that flows quietly through the city.",
        "She carefully threads the needle, her hands steady as she begins to sew the intricate pattern on the delicate fabric, creating a beautiful masterpiece.",
        "The aroma of freshly brewed coffee fills the air as he sits at the table, slowly stirring sugar into his cup while watching the sunrise over the horizon."
    ]

    #Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    #Get the gmm and scalers for each phrase and append to a global gmma and scalar
    for person in range(num_people):
        print(f"Training person {person + 1}/{num_people}")
        gmm, scaler = train_gmm_for_person(phrases)
        gmms.append(gmm)
        scalers.append(scaler)

    #Once all voices are trained, predict by recordig
    while True:
        predict_voice(gmms, scalers)
        time.sleep(3)


if __name__ == "__main__":
    main()
