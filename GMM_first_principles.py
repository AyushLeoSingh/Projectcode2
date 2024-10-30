import numpy as np
import os
import librosa
import librosa.display
import sounddevice as sd
from sklearn.preprocessing import StandardScaler
import time
from scipy import linalg, sparse
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.mixture import GaussianMixture

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







def extract_mfcc_from_wav(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs


def load_wav_files(directory):
    mfcc_list = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".wav"):
            file_path = os.path.join(directory, file_name)
            mfccs = extract_mfcc_from_wav(file_path)
            mfcc_list.append(mfccs)
    return mfcc_list


def train_GMM_waves(directory):
    # Load all wav files and extract MFCCs
    mfcc_list = load_wav_files(directory)

    # Combine MFCCs from all files into one array
    combined_mfccs = np.hstack(mfcc_list)  # Concatenate along the second axis

    # No scaling is performed
    mfccs_transposed = combined_mfccs.T

    # Train the UBM GMM model
    gmm = GMM(n_components=16, max_iter=200)
    print("GMM Model created")
    gmm.fit(mfccs_transposed)
    print("GMM UBM Model trained")

    print("GMM UBM Training Complete using .wav files.")
    return gmm

def load_training_data(filename):
    mfccs_transposed = np.loadtxt(filename, delimiter=',')
    print("Training data loaded from", filename)
    return mfccs_transposed


def get_smoothed_data(serial_port='COM13', baud_rate=115200, record_seconds=5,
                      low_cutoff_freq=100, high_cutoff_freq=700):
    SAMPLING_RATE = 44100
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

    audio_data = audio_data.flatten()
    baseline = np.mean(audio_data)
    audio_data -= baseline

    return audio_data, SAMPLING_RATE


def capture_audio(duration=5, sampling_rate=44100):
    smoothed_audio_data, sampling_rate = get_smoothed_data(record_seconds=duration)
    return smoothed_audio_data, sampling_rate

def train_GMM(Name):
    combined_mfccs = load_training_data(Name)
    scaler = StandardScaler()
    mfccs_normalized = scaler.fit_transform(combined_mfccs.T).T
    mfccs_transposed = mfccs_normalized.T

    gmm = GMM(n_components=16, max_iter=100, covariance_type="diag")
    print("created")
    gmm.fit(mfccs_transposed)
    print("trained")

    print("GMM Training Complete using loaded data.")
    return gmm, scaler


def predict_voice(gmms, scalers , duration=5, sampling_rate=44100):
    # Capture audio
    waveform, sample_rate = capture_audio(duration, sampling_rate)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)

    # Initialize variables to track the best match
    best_person = None
    highest_likelihood = -np.inf
    total_liklihoods=[]

    for idx, (gmm, scaler) in enumerate(zip(gmms, scalers)):
        mfccs_normalized = scaler.transform(mfccs.T).T
        mfccs_transposed = mfccs_normalized.T
        # Compute the log likelihood directly on raw MFCC features
        log_likelihood = gmm.score(mfccs.T)

        # Calculate the average log likelihood
        avg_likelihood = np.mean(log_likelihood)
        print(f"GMM {idx + 1} likelihood: {avg_likelihood}")
        total_liklihoods.append(avg_likelihood/10000)

        # Determine the most likely GMM
        if avg_likelihood > highest_likelihood:
            highest_likelihood = avg_likelihood
            best_person = idx + 1

    total_liklihoods=np.array(total_liklihoods)
    #print(f"Voice is most similar to GMM {best_person}")
    print("Averaged is: ", np.mean(total_liklihoods))
    if(np.mean(total_liklihoods)>=-135 and np.mean(total_liklihoods)<=-110):
        print("Voice Detected")
    else:
        print("Voice not Detected")

def main():
    # Arrays for the GMM and Scalers
    gmms = []
    scalers = []

    gmm, scaler = train_GMM("Ayush_TestingData3.txt")
    gmms.append(gmm)
    scalers.append(scaler)
    gmm, scaler = train_GMM("Ayush_TestingData4.txt")
    gmms.append(gmm)
    scalers.append(scaler)
    gmm, scaler = train_GMM("Ayush_TestingData5.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Mo_TestingData4.txt")
    gmms.append(gmm)
    scalers.append(scaler)
    gmm, scaler = train_GMM("Mo_TestingData5.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Rub_TestingData2.txt")
    gmms.append(gmm)
    scalers.append(scaler)
    gmm, scaler = train_GMM("Rub_TestingData3.txt")
    gmms.append(gmm)
    scalers.append(scaler)
    gmm, scaler = train_GMM("Rub_TestingData5.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Ayush1.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Ayush_Data3.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Kim_Data.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Vidic_Data.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    gmm, scaler = train_GMM("Sejal_Data.txt")
    gmms.append(gmm)
    scalers.append(scaler)

    # Train using .wav files (newly added for UBM)
    #ubm_gmm = train_GMM_waves(r"C:\Users\Ayush\Desktop\EPR code\Lib\New_UBM")  # Path to directory with .wav files
    #gmms.append(ubm_gmm)  # Add UBM GMM to the list

    # Once all voices are trained, predict by recording
    while True:
        predict_voice(gmms,scalers)  # Pass the list of GMMs
        time.sleep(3)


if __name__ == "__main__":
    main()