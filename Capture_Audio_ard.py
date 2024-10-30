import serial
import time
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np
from scipy.signal import butter, filtfilt

# Set up the serial connection (adjust 'COM13' to your port)
ser = serial.Serial('COM13', 1000000)  # Set to match the baud rate used in Arduino
sample_rate = 20000  # Sampling rate (set as needed)
duration = 5  # Record for 5 seconds
num_samples = 25  # Number of samples to read in each cycle
audio_data = []
total_samples_received = 0  # Counter for total samples received


def highpass_filter(data, cutoff, fs, order=5):
    # Design a high-pass filter using the Butterworth method
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    # Apply the filter to the data
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')


    return filtfilt(b, a, data)


start_time = time.time()

# Record audio for the specified duration
print("Recording...")
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

cutoff_frequency = 100  # Choose the appropriate cutoff (e.g., 100 Hz)
sample_rate = 21000  # Replace with your actual sample rate
filtered_audio = highpass_filter(audio_array, cutoff_frequency, sample_rate)

lowcut = 40
highcut = 1700
filtered_audio = bandpass_filter(audio_array, lowcut, highcut, sample_rate)
for i in range(len(filtered_audio)):
    filtered_audio[i]*=7

# Play back the gated audio
sd.play(audio_array, samplerate=sample_rate)
sd.wait()  # Wait until playback is done
sd.play(filtered_audio, samplerate=22000)
sd.wait()  # Wait until playback is done




# Plot the original and filtered waveforms
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(audio_array, color='blue')
plt.title('Original Waveform')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(filtered_audio, color='blue')
plt.title('Original Waveform')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# Perform FFT on the audio data
fft_result = np.fft.fft(filtered_audio)
frequencies = np.fft.fftfreq(len(fft_result), 1/sample_rate)

# Get the magnitude of the FFT result
magnitude = np.abs(fft_result)

# Plot the frequency spectrum (up to Nyquist frequency)
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2], color='blue')
plt.title('Frequency Spectrum of the Audio Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Close the serial port
ser.close()
