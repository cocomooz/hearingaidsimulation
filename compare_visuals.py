import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def plot_spec(file, title, idx):
    rate, data = wav.read(file)
    plt.subplot(3, 1, idx)
    plt.specgram(data, Fs=rate, NFFT=1024, noverlap=512, cmap='magma')
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Freq [Hz]")

def main():
    files = ["noisy_mixture.wav", "standard_output.wav", "iss_output.wav"]
    titles = ["Noisy Input", "Standard Spectral Subtraction", "Improved SS (ISS)"]

    plt.figure(figsize=(12, 10))
    for i, (file, title) in enumerate(zip(files, titles)):
        plot_spec(file, title, i + 1)

    plt.tight_layout()
    plt.savefig("compare_spectrograms.png")
    plt.show()
    print("Saved comparison image to 'compare_spectrograms.png'")

if __name__ == "__main__":
    main()
