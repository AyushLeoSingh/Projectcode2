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
import serial
from scipy.spatial.distance import cdist

#Use to test
#Voice Class works, GMM from First Principles
#MFCCs are not from First Principles
#Deltas added
#
MicOption = True
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


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, data)







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


def get_smoothed_data_ard(serial_port='COM13', baud_rate=115200, record_seconds=5,
                      low_cutoff_freq=100, high_cutoff_freq=700):
    ser = serial.Serial('COM13', 1000000)  # Set to match the baud rate used in Arduino
    duration = 5  # Record for 5 seconds
    num_samples = 25  # Number of samples to read in each cycle
    audio_data = []
    total_samples_received = 0  # Counter for total samples received

    SAMPLING_RATE = 20000

    print("Start")
    time.sleep(1)
    print("Recording...")
    time.sleep(1)
    start_time = time.time()
    while time.time() - start_time < duration:
        if ser.in_waiting >= num_samples * 2:  # Check if enough bytes are available (2 bytes per sample)
            # Read the binary data directly
            data = ser.read(num_samples * 2)  # Read the specified number of samples
            audio_data.extend(
                np.frombuffer(data, dtype=np.uint16))  # Convert binary data to numpy array and extend audio_data
            total_samples_received += num_samples  # Update the sample counter

    # Print the total number of samples received
    print(f"Total samples received: {total_samples_received}")

    # Convert the audio data to a NumPy array for playback
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array *= 13  # Amplification factor, adjust as necessary
    audio_array = np.interp(audio_array, (0, 4095), (-1, 1))  # Normalize ADC values (0-4095) to (-1, 1)

    sample_rate = 20000  # Replace with your actual sample rate

    lowcut = 40
    highcut = 1700
    filtered_audio = bandpass_filter(audio_array, lowcut, highcut, sample_rate)
    for i in range(len(filtered_audio)):
        filtered_audio[i] *= 7

    return filtered_audio, SAMPLING_RATE


def capture_audio(duration=5, sampling_rate=44100):
    smoothed_audio_data, sampling_rate = get_smoothed_data(record_seconds=duration)
    return smoothed_audio_data, sampling_rate

def capture_audio_ard(duration=5, sampling_rate=44100):
    smoothed_audio_data, sampling_rate = get_smoothed_data_ard(record_seconds=duration)
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

    if arrlikelyhoods[0]>-27:
        print("Silence")
    if arrlikelyhoods[0]<-40 and arrlikelyhoods[1]>-20:
        print("Window broken")

def pretrained_usb(mfccs):
    # Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    # Load the stored voices
    gmm, scaler = train_GMM("sejal1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("sejal2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("sejal3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mathew3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rubin3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Johan3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Wisdom3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Andre1_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Andre2_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Andre3_usb.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    averages = []
    arrlikelyhoods = []
    highest_likelihood = -np.inf

    count = 0
    index = 0
    # Compare the recorded voice to the stored voice
    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        likelihood = gmm.score(mfccs_transposed)
        avg_likelihood = np.mean(likelihood)

        print(f"Person {idx + 1} likelihood: {avg_likelihood}")

        averages.append(avg_likelihood)
        arrlikelyhoods.append(avg_likelihood)
        if avg_likelihood > highest_likelihood:
            index = count
            highest_likelihood = avg_likelihood
            best_person = idx + 1
        count += 1

    sej_avg = np.mean(averages[0:3])
    mat_avg = np.mean(averages[3:6])
    rub_avg = np.mean(averages[6:9])
    johan_avg = np.mean(averages[9:12])
    wisdom_avg = np.mean(averages[12:15])
    andre_avg = np.mean(averages[15:])

    print("avgs: ", sej_avg, mat_avg, rub_avg, johan_avg, wisdom_avg, andre_avg)
    average_array = np.array([sej_avg, mat_avg, rub_avg, johan_avg, wisdom_avg, andre_avg])
    min_index = np.argmax(average_array)
    # print("min_index: ",min_index)

    # If the person is on the system then match
    # Else no match
    if (highest_likelihood > -20):
        print("Voice in system")
        if (index < 3 and min_index == 0):
            print("Person 1 identified")
        if (index > 2 and index < 6 and min_index == 1):
            print("Person 2 identified")
        if (index > 5 and index < 9 and min_index == 2):
            print("Person 3 identified")
        if (index > 8 and index < 12 and min_index == 3):
            print("Person 4 identified")
        if (index > 11 and index < 15 and min_index == 4):
            print("Person 5 identified")
        if (index > 15 and index < 19 and min_index == 5):
            print("Person 6 identified")

    else:
        print("Voice not in system")


def pretrained(mfccs):
    # Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    #Load the stored voices
    gmm, scaler = train_GMM("s1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("s2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("s3_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("m1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("m2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("m3_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("r1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("r2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("r3_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("j1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("j2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("j3_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("w1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("w2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("w3_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("andre1_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("andre2_ard.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("a3_ard.txt")
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

    sej_avg = np.mean(averages[0:3])
    mat_avg = np.mean(averages[3:6])
    rub_avg = np.mean(averages[6:9])
    johan_avg = np.mean(averages[9:12])
    wisdom_avg = np.mean(averages[12:15])
    andre_avg = np.mean(averages[15:])

    print("avgs: ", sej_avg, mat_avg, rub_avg, johan_avg, wisdom_avg, andre_avg)
    average_array = np.array([sej_avg, mat_avg, rub_avg, johan_avg, wisdom_avg, andre_avg])
    min_index = np.argmax(average_array)
    #print("min_index: ",min_index)

        #If the person is on the system then match
        #Else no match
    if(highest_likelihood>-20):
        print("Voice in system")
        if (index<3 and min_index==0):
            print("Person 1 identified")
        if (index>2 and index<6 and min_index==1):
            print("Person 2 identified")
        if (index>5 and index <9 and min_index==2):
            print("Person 3 identified")
        if (index>8 and index <12 and min_index==3):
            print("Person 4 identified")
        if (index>11 and index<15 and min_index==4):
            print("Person 5 identified")
        if (index>15 and index<19 and min_index==5):
            print("Person 6 identified")

    else:
        print("Voice not in system")



def Classify_as_voice(gmms, scalers, duration=5, sampling_rate=21000):
    averages=[]
    threshold_db = 70
    waveform, sample_rate = capture_audio(duration, sampling_rate)
    lowcut = 40
    highcut = 1700
    filtered_audio = bandpass_filter(waveform, lowcut, highcut, sampling_rate)
    waveform = filtered_audio

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
        #print(f"Person {idx + 1} likelihood: {avg_likelihood}")
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
    #print("Averages: ", averages)

    male_avg = np.mean(averages[0:3])
    female_avg = np.mean(averages[3:5])
    sej_avg = np.mean(averages[5:8])
    mat_avg = np.mean(averages[8:11])
    rub_avg = np.mean(averages[11:14])
    johan_avg = np.mean(averages[14:17])
    wisdom_avg = np.mean(averages[17:20])
    andre_avg = np.mean(averages[20:])

    print("male_avg: ", male_avg)
    print("female_avg: ", female_avg)
    print("sej_avg: ",sej_avg)
    print("mat_avg: ", mat_avg)
    print("rub_avg: ", rub_avg)
    print("johan_avg: ", johan_avg)
    print("wisdom_avg: ", wisdom_avg)
    print("andre_avg: ", andre_avg)

    train=False
    if(male_avg>-20 or female_avg>-20 or sej_avg>-22.5 or mat_avg>-22.5 or rub_avg>-22.5 or johan_avg>-22.5 or wisdom_avg>-22.5 or andre_avg>-22.5) and silenceavg<-50:
        train=True
        print("Voice Detected!")


    if train==False:
        gmmsx = []
        scalersx = []
        gmm, scaler = train_GMM("silence20000_usb.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("window20000_usb.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("plates20000_usb.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("wineglass20000_usb.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("wood20000_usb.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)


        Classify_as_breakage(mfccs, gmmsx, scalersx)

    if train==True:
        pretrained_usb(mfccs)


def Classify_as_voice_ard(gmms, scalers, duration=5, sampling_rate=20000):
    averages=[]
    threshold_db = 90
    waveform, sample_rate = capture_audio_ard(duration, sampling_rate)

    new_waveform=section_waveform(waveform,10000)

    # Calculate SPL
    spl_value = calculate_spl_2d(new_waveform)

    #print(f"SPL: {spl_value:.2f} dB")

    if spl_value > threshold_db:
        print(f"Warning: Detected SPL is above 70 dB!")
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
    male_avg = np.mean(averages[0:3])
    female_avg = np.mean(averages[3:6])
    sej_avg = np.mean(averages[6:9])
    mat_avg = np.mean(averages[9:12])
    rub_avg = np.mean(averages[12:15])
    johan_avg = np.mean(averages[15:18])
    wisdom_avg = np.mean(averages[18:21])
    andre_avg = np.mean(averages[21:24])

    print("male_avg: ", male_avg)
    print("female_avg: ", female_avg)
    print("sej_avg: ", sej_avg)
    print("mat_avg: ", mat_avg)
    print("rub_avg: ", rub_avg)
    print("johan_avg: ", johan_avg)
    print("wisdom_avg: ", wisdom_avg)
    print("andre_avg: ", andre_avg)

    train = False
    if (male_avg > -23 or female_avg > -23 or sej_avg >-22 or mat_avg>-22 or rub_avg>-22 or johan_avg>-22 or wisdom_avg>-22 or andre_avg>-22) and silenceavg < -35:
        train = True
        print("Voice Detected!")


    if train == False:
        gmmsx = []
        scalersx = []
        gmm, scaler = train_GMM("silence20000ard.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        gmm, scaler = train_GMM("window_ard.txt")
        gmmsx.append(gmm)
        scalersx.append(scaler)

        Classify_as_breakage(mfccs, gmmsx, scalersx)

    if train == True:
        pretrained(mfccs)




def main():
    ans=input("Use USB Mic? Y/N")
    if ans.upper()=='N':
        MicOption=False
    else:
        MicOption=True


    if MicOption==True:
        #Arrays for the GMM and Scalers
        gmms = []
        scalers = []

        gmm, scaler = train_GMM("silence20000_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("nishlin20000_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("reagan20000_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Ayush20000filt.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("kim20000_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("sejal20000_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("sejal1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("sejal2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("sejal3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Mathew1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Mathew2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Mathew3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Rubin1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Rubin2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Rubin3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Johan1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Johan2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Johan3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Wisdom1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Wisdom2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Wisdom3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Andre1_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Andre2_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Andre3_usb.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        while True:
            train=False
            Classify_as_voice(gmms, scalers)
            time.sleep(3)

    if MicOption==False: #Use serial comms
        # Arrays for the GMM and Scalers
        gmms = []
        scalers = []
        gmm, scaler = train_GMM("silence20000ard2.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("Ayush20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("reagan20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("nishlin20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("kim20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("sejal20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("hatalya20000ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("s1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("s2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("s3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("m1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("m2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("m3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("r1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("r2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("r3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("j1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("j2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("j3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("w1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("w2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("w3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("a1_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("a2_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)

        gmm, scaler = train_GMM("a3_ard.txt")
        gmms.append(gmm)
        scalers.append(scaler)


        while True:
            train=False
            Classify_as_voice_ard(gmms, scalers)
            time.sleep(3)


if __name__ == "__main__":
    main()
