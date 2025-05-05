import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

FRAME_SIZE = 1024
OVERLAP = 512
ALPHA = 0.95

def match_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]

# --- Standard Spectral Subtraction ---
def standard_spectral_subtraction(signal_in, fs):
    _, _, Zxx = signal.stft(signal_in, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    noise_est = np.copy(magnitude[:, 0])

    for i in range(1, magnitude.shape[1]):
        noise_est = ALPHA * noise_est + (1 - ALPHA) * magnitude[:, i]
        magnitude[:, i] = np.maximum(magnitude[:, i] - noise_est, 0)

    processed = magnitude * np.exp(1j * phase)
    _, audio_out = signal.istft(processed, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    return audio_out


# --- SNR Calculation ---
def compute_snr(clean, enhanced):
    clean, enhanced = match_length(clean, enhanced)
    noise = clean - enhanced
    snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))
    return snr


# --- Main Runner ---
def main():
    fs, audio = wav.read("noisy_mixture.wav")
    audio = audio.astype(np.float32)

    _, clean = wav.read("clean_speech.wav")
    clean = clean.astype(np.float32)
    if clean.ndim > 1:
        clean = clean[:, 0]

    denoised = standard_spectral_subtraction(audio, fs)
    denoised = np.clip(denoised, -32768, 32767).astype(np.int16)

    wav.write("standard_output.wav", fs, denoised)
    print("Saved standard spectral subtraction result to 'standard_output.wav'")

    snr_val = compute_snr(clean[:len(denoised)], denoised)
    print(f"SNR (Standard Spectral Subtraction): {snr_val: .2f} dB")


if __name__ == "__main__":
    main()
