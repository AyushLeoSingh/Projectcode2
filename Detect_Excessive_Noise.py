import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import librosa.display


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

    audio_data = audio_data.flatten()
    baseline = np.mean(audio_data)
    audio_data -= baseline

    return audio_data, SAMPLING_RATE


def calculate_spl(waveform):
    # Calculate RMS of the waveform
    rms = np.sqrt(np.mean(waveform ** 2))

    # Reference sound pressure in air (in Pa)
    p0 = 20e-6

    # Calculate SPL
    spl = 20 * np.log10(rms / p0)

    return spl


def capture_audio(duration=5, sampling_rate=2500):
    smoothed_audio_data, sampling_rate = get_smoothed_data(record_seconds=duration)
    return smoothed_audio_data, sampling_rate


def detect_excessive_noise(duration=5, sampling_rate=2500, threshold_db=60):
    waveform, sample_rate = capture_audio(duration, sampling_rate)

    # Calculate SPL
    spl_value = calculate_spl(waveform)

    print(f"SPL: {spl_value:.2f} dB")

    if spl_value > threshold_db:
        print(f"Warning: Detected SPL is above {threshold_db} dB!")
    else:
        print(f"Detected SPL is below {threshold_db} dB.")

    plot_waveform(waveform, sample_rate)


def plot_waveform(waveform, sampling_rate):
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


def main():
    while True:
        detect_excessive_noise()


if __name__ == "__main__":
    main()
