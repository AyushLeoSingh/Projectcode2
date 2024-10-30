import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, savgol_filter
import time
from scipy.special import logsumexp
from sklearn import cluster
from scipy import linalg
from scipy.spatial.distance import cdist

#Use to test
#Voice Class works, GMM from First Principles
#MFCCs are not from First Principles
#Deltas added
#
stored_voices=[]

####################################################GMM#################################################################
########################################################################################################################
def est_covariances_full(resp, X, nk, means, reg_cov):
    n_components, n_features = means.shape
    cov = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        cov[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        cov[k].flat[:: n_features + 1] += reg_cov
    return cov

def est_covariances_diag(resp, X, nk, means, reg_cov):
    avg_X2 = np.dot(resp.T, X * X) / nk[:, np.newaxis]
    avg_means2 = means**2
    avg_X_means = means * np.dot(resp.T, X) / nk[:, np.newaxis]
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_cov

def est_params(X, resp, reg_covar, covariance_type):
    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]

    if covariance_type == 'full':
        covariances = est_covariances_full(resp, X, nk, means, reg_covar)
    else:
        covariances = est_covariances_diag(resp, X, nk, means, reg_covar)
    return nk, means, covariances

def calc_prec_chol(covariances, covariance_type):
    if covariance_type == "full":
        n_components, n_features, _ = covariances.shape
        cholsky = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            cov_chol = linalg.cholesky(covariance, lower=True)
            cholsky[k] = linalg.solve_triangular(
                cov_chol, np.eye(n_features), lower=True
            ).T
    else:
        cholsky = 1.0 / np.sqrt(covariances)
    return cholsky


def calc_log_det_chol(matrix_chol, covariance_type, n_features):
    if covariance_type == "full":
        n_components, _, _ = matrix_chol.shape
        log_det_chol = np.sum(
            np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )


    else:
        log_det_chol = np.sum(np.log(matrix_chol), axis=1)


    return log_det_chol

def e_log_prob(X, means, prec_chol, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    log_det = calc_log_det_chol(prec_chol, covariance_type, n_features)

    if covariance_type == "full":
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, prec_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)


    else:
        precisions = prec_chol ** 2
        log_prob = (
            np.sum((means**2 * precisions), 1)
            - 2.0 * np.dot(X, (means * precisions).T)
            + np.dot(X**2, precisions.T)
        )

    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class GMM:
    def __init__(self, n_components, max_iter=5, weights_init=None,means_init=None, precisions_init=None, covariance_type="diag",):
        self.n_components = n_components
        self.max_iter = int(max_iter)
        self.cov_type = str(covariance_type), #diag is default as most sound data is diagonal
        self.breakval = 1e-3,
        self.reg_cov = 1e-6 #tol and reg_covar are same as sklearns

    def initialize(self, X, random_state):
        n_samples, _ = X.shape
        resp = np.zeros((n_samples, self.n_components))
        label = (
            cluster.KMeans(
                n_clusters=self.n_components, n_init=1, random_state=random_state
            )
            .fit(X)
            .labels_
        )
        resp[np.arange(n_samples), label] = 1

        weights, means, covariances = est_params(
            X, resp, self.reg_cov, self.cov_type[0]
        )

        weights /= n_samples

        self.weights_ = weights
        self.means_ = means

        self.covariances_ = covariances
        self.prec_chol = calc_prec_chol(
            covariances, self.cov_type[0]
        )
    def fit(self, X):
        max_lower_bound = -np.inf
        self.conv = False

        random_state = np.random.mtrand._rand # random state


        n_samples, _ = X.shape


        self.initialize(X, random_state)

        lower_bound = -np.inf



        for i in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self.e_step(X)
            self.m_step(X, log_resp)
            lower_bound = log_prob_norm

            change = lower_bound - prev_lower_bound

            if abs(change) < self.breakval:
                self.converged_ = True
                break


        self.lower_bound_ = max_lower_bound



    def e_step(self,X):
        #The expectation step calculculates the expected log-likelyhood using the current parameters
        weighted_log_prob = e_log_prob(X, self.means_, self.prec_chol,
                                       self.cov_type[0]) + np.log(self.weights_)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return np.mean(log_prob_norm), log_resp


    def m_step(self, X, log_resp):
        #the maximization step adjusts the para,eters based on the results of the e step.
        self.weights_, self.means_, self.covariances_ = est_params(
            X, np.exp(log_resp), self.reg_cov, self.cov_type[0]
        )
        self.weights_ /= self.weights_.sum()
        self.prec_chol = calc_prec_chol(
            self.covariances_, self.cov_type[0]
        )
    def predict(self,X):
        #This is used mostly for testing but it determine which cluster the data X best fits into
        weighted_log_prob = e_log_prob(X, self.means_, self.prec_chol, self.cov_type[0]) + np.log(self.weights_)


        return weighted_log_prob.argmax(axis=1)

    def score(self,X):
        #Calculates the log likelyhood of thd given data X. This is what is used to calculate the
        weighted_log_prob = e_log_prob(X, self.means_, self.prec_chol,
                                       self.cov_type[0]) + np.log(self.weights_)
        log_sum = logsumexp(weighted_log_prob,axis=1)

        return log_sum.mean()
########################################################################################################################
###########################################END GMM######################################################################











#############################################Deltas#####################################################################

def calculate_delta(array, N=2):
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))  # Initialize delta array with the same shape

    for i in range(rows):
        numerator = np.zeros(cols)
        denominator = 0

        for n in range(1, N + 1):
            # Handle boundaries by mirroring the index
            next_index = min(i + n, rows - 1)
            prev_index = max(i - n, 0)

            # Calculate the weighted difference
            numerator += n * (array[next_index] - array[prev_index])
            denominator += 2 * (n ** 2)

        deltas[i] = numerator / denominator

    return deltas
##########################################Deltas########################################################################









train=False
new_gmms=[]
new_scalers=[]
#This script has excessive noise detection and voice classificstion on it

def calculate_spl(waveform):
    # Calculate RMS of the waveform
    rms = np.sqrt(np.mean(waveform ** 2))

    # Reference sound pressure in air (in Pa)
    p0 = 20e-6

    # Calculate SPL
    spl = 20 * np.log10(rms / p0)

    return spl


def calculate_spl_2d(waveform_2d):
    # Initialize the highest SPL value
    highest_spl = -np.inf

    # Iterate over each row in the 2D array
    for row in waveform_2d:
        # Calculate RMS of the row (waveform)
        rms = np.sqrt(np.mean(row ** 2))

        # Reference sound pressure in air (in Pa)
        p0 = 20e-6

        # Calculate SPL for the current row
        spl = 20 * np.log10(rms / p0)

        # Update the highest SPL if the current one is higher
        if spl > highest_spl:
            highest_spl = spl

    return highest_spl

def load_training_data(filename):
    mfccs_transposed = np.loadtxt(filename, delimiter=',')
    #print("Training data loaded from", filename)
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
    #Voice saved


    audio_data = audio_data.flatten()
    baseline = np.mean(audio_data)
    audio_data -= baseline
    #audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)


    N = len(audio_data)
    sampling_rate = 1 / (record_seconds / N)
    if low_cutoff_freq >= 0.5 * sampling_rate or high_cutoff_freq >= 0.5 * sampling_rate:
        raise ValueError("Cutoff frequencies must be less than half the sampling rate.")

    #Filter the audio to reduce noise
    filtered_audio_data = butter_bandpass_filter(audio_data, low_cutoff_freq, high_cutoff_freq, sampling_rate)


    #Smoothen the data using savgol
    #window_length = 51
    #polyorder = 3
    #smoothed_audio_data = savgol_filter(filtered_audio_data, window_length, polyorder)
    #smoothed_audio_data = np.interp(smoothed_audio_data, (smoothed_audio_data.min(), smoothed_audio_data.max()),
    #                                (filtered_audio_data.min(), filtered_audio_data.max()))
    #smoothed_audio_data = smoothed_audio_data[10:len(smoothed_audio_data) - 10]

    return audio_data, SAMPLING_RATE


def capture_audio(duration=5, sampling_rate=44100):
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


def section_waveform(waveform, N):
    # Calculate the size of each section
    section_size = len(waveform) // N
    # Create an empty 2D array to store the sections
    result = np.zeros((N, section_size))

    # Fill the 2D array with sections of the waveform
    for i in range(N):
        start_idx = i * section_size
        end_idx = start_idx + section_size
        result[i] = waveform[start_idx:end_idx]

    return result


def train_GMM(Name):
    combined_mfccs = load_training_data(Name)

    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    #gmm = GaussianMixture(n_components=10, covariance_type='diag'
    gmm = GMM(n_components=10, max_iter=100, covariance_type='diag')
    gmm.fit(mfccs_transposed)

    #print("GMM Training Complete using loaded data.")
    return gmm, scaler


from scipy.spatial.distance import cdist


def Compare_Voices(stored_mfccs, new_mfccs, threshold=490, likelihood_threshold=-30.5):

    #Get the GMM and Scalar for the stored voice
    combined = np.hstack(stored_mfccs).reshape(-1, 1)
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined.T).T
    mfccs_transposed = mfccs_normalized.T
    gmm = GMM(n_components=10, max_iter=100, covariance_type='diag')
    gmm.fit(mfccs_transposed)


    #Get the mfccs of the new voice and compare to the stored one
    mfccs_normalized = scaler.transform(new_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T
    likelihood = gmm.score(mfccs_transposed)
    avg_likelihood=np.mean(likelihood)

    #print("likelihood: ",avg_likelihood)

    # Return True if both checks pass
    if(avg_likelihood>-50):
        return True
    else:
        return False


def Classify_as_breakage(mfccs, gmms, scalers):
    averages = []
    highest_likelihood = -np.inf
    arrlikelyhoods = []

    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)
        print(f"Person {idx + 1} likelihood: {avg_likelihood}")
        averages.append(avg_likelihood)
        arrlikelyhoods.append(avg_likelihood)

    '''
    if arrlikelyhoods[0]<-35: #if not silence
        if arrlikelyhoods[0]>-60 and arrlikelyhoods[2]>-19:
            print("Wine Glass Broken")
        if arrlikelyhoods[0]<-100 and arrlikelyhoods[1]>arrlikelyhoods[2] and arrlikelyhoods[1]>arrlikelyhoods[3] and arrlikelyhoods[1]>-30:
            print("Window Broken")
        if arrlikelyhoods[3]>arrlikelyhoods[2] and arrlikelyhoods[3]>arrlikelyhoods[1] and arrlikelyhoods[0]>-98:
            print("Plate Broken")
    else:
        print("Silence")
    '''
    plate=False
    if arrlikelyhoods[0]>-89:
        print("Silence")
    if arrlikelyhoods[0]>-120 and arrlikelyhoods[0]<-80 and arrlikelyhoods[4]>-35:
        print("Woods broken")
    if arrlikelyhoods[0]<-110 and arrlikelyhoods[3]>-36 and arrlikelyhoods[3]>arrlikelyhoods[2]:
        print("Plates broken")
        plate=True
    if arrlikelyhoods[0] < -110 and arrlikelyhoods[3] > -39 and arrlikelyhoods[2] > -39 and plate==False:
        print("Wine glass broken")
    if arrlikelyhoods[0]<-180 and arrlikelyhoods[1]>-39:
        print("Window Broken")
def pretrained(mfccs):
    # Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    #Load the stored voices
    gmm, scaler = train_GMM("Sejal_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    averages=[]
    arrlikelyhoods=[]
    highest_likelihood = -np.inf

    count=0
    index=0
    #Compare the recorded voice to the stored voice
    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)

        print(f"Person {idx + 1} likelihood: {avg_likelihood}")

        averages.append(avg_likelihood)
        arrlikelyhoods.append(avg_likelihood)
        if avg_likelihood > highest_likelihood:
            index=count
            highest_likelihood = avg_likelihood
            best_person = idx + 1
        count+=1

        #If the person is on the system then match
        #Else no match
    if(highest_likelihood>-22):
        print("Voice in system")
        if index<3:
            print("Person 1 identified")
        if (index>2 and index<6):
            print("Person 2 identified")
        if (index>5 and index <9):
            print("Person 3 identified")
        if (index>8 and index <12):
            print("Person 4 identified")
        if (index>11 and index<15):
            print("Person 5 identified")
        if (index>14 and index<18):
            print("Person Mo identified")

    else:
        print("Voice not in system")



def Classify_as_voice(gmms, scalers, duration=5, sampling_rate=44100):
    averages=[]
    threshold_db = 70
    waveform, sample_rate = capture_audio(duration, sampling_rate)

    new_waveform=section_waveform(waveform,1000)

    # Calculate SPL
    spl_value = calculate_spl_2d(new_waveform)

    print(f"SPL: {spl_value:.2f} dB")

    if spl_value > threshold_db:
        print(f"Warning: Detected SPL is above {threshold_db} dB!")
        print("Excessive Noise Detected!")
    else:
        print("Noise levels are fine")


    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

    best_person = None
    highest_likelihood = -np.inf
    arrlikelyhoods = []

    count=0
    silenceavg=0
    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)
        print(f"Person {idx + 1} likelihood: {avg_likelihood}")
        if count==0:
            silenceavg=avg_likelihood
        #Silence is first, dont use it for calculations
        if count>0:
            averages.append(avg_likelihood)
            arrlikelyhoods.append(avg_likelihood)
            if avg_likelihood > highest_likelihood:
                highest_likelihood = avg_likelihood
                best_person = idx + 1
        count+=1
    print("Silence score is: ", silenceavg)


    detected=0
    train=False
    male_avg = np.mean(averages[0:4])
    female_avg = np.mean(averages[4:9])
    sej_avg = np.mean(averages[9:12])
    mathew_avg = np.mean(averages[12:15])
    rubin_avg = np.mean(averages[15:18])
    johan_avg = np.mean(averages[18:21])
    wisdom_avg = np.mean(averages[21:24])
    mo_avg = np.mean(averages[24:27])

    window_avg = averages[27]

    #This checks for the highest of the people in the system to demo
    highest_score = max(averages[10:28])



    avg=np.mean(averages)

    print("avg is: ", avg)
    print("m avg is: ", female_avg)
    print("f avg is: ", male_avg)
    print("sej avg is: ", sej_avg)
    print("mathew avg is: ", mathew_avg)
    print("rubin avg is: ", rubin_avg)
    print("johan avg is: ", johan_avg)
    print("wisdom avg is: ", wisdom_avg)
    print("mo avg is: ", mo_avg)
    print("Highest score", highest_score)
    print("Window score: ", window_avg)

    if ((avg > -30 or female_avg>-30 or male_avg>-30 or sej_avg>-25 or mathew_avg>-25 or rubin_avg>-25 or johan_avg>-25 or wisdom_avg>-25 or mo_avg>-30 or highest_score>-20) and silenceavg<-130 and window_avg<-35):
        print("Voice Detected!")
        train=True
    else:
        print("Voice not Detected.")

    if train==False:
        gmmsx = []
        scalersx = []
        gmm, scaler = train_GMM("Lab_silence.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("Lab_Window.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("Wineglass_lab.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("Plates_lab.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("Sticks_lab.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)


        Classify_as_breakage(mfccs, gmmsx, scalersx)

    if train==True:
        pretrained(mfccs)
        '''
        print("Training")
        # Extract MFCCs for comparison
        new_mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        if len(stored_voices)==0:
            print("First Voice Detected")
            stored_voices.append(new_mfccs)
        else:
            print("Not the first voice")
            # Compare with stored voices
            voice_matched = False
            for stored_voice in stored_voices:
                if Compare_Voices(stored_voice, new_mfccs):
                    print("Voice matched with stored voice.")
                    voice_matched = True
                    break
            if not voice_matched:
                print("no match with stored voices")
                # Save the new voice data
                stored_voices.append(new_mfccs)
        '''


def main():
    #Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    gmm, scaler = train_GMM("Lab_silence.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Ayush_Unfitered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Ayush25000.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Koot_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Kim_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Hatalya_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Matshepo_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Aclaire_Unfiltered.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_2.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_stored_3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Lab_Window.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    while True:
        train=False
        Classify_as_voice(gmms, scalers)
        time.sleep(3)


if __name__ == "__main__":
    main()
