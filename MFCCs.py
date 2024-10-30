import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_mel_filter_bank_with_triangles(sr=22050, n_fft=2048, n_mels=10):
    # Generate the Mel filter bank
    mel_filter_bank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # Plot each triangular filter
    plt.figure(figsize=(10, 6))
    for i in range(n_mels):
        plt.plot(mel_filter_bank[i], label=f'Filter {i+1}')
        print(mel_filter_bank[i])

    # Add labels and titles
    plt.title('Mel Filter Bank with Triangular Filters')
    plt.xlabel('FFT Bin')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
plot_mel_filter_bank_with_triangles(sr=22050, n_fft=2048, n_mels=10)
