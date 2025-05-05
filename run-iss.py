#method proposed by Liu et al.,
"""
Liu, Q., Yu, Y., Han, B. S., & Zhou, W. (2024).
An Improved Spectral Subtraction Method for Eliminating
Additive Noise in Condition Monitoring System Using Fiber
Bragg Grating Sensors. Sensors, 24(2), 443.
doi.org/10.3390/s24020443.
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal

# --- Parameters ---
FRAME_SIZE = 1024
OVERLAP = 512
THRESHOLD_PEAK_RATIO = 0.5

# --- Load audio ---
def load_audio(file):
    rate, data = wav.read(file)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return rate, data.astype(np.float32)

# --- STFT ---
def compute_spectrogram(audio, fs):
    if OVERLAP >= FRAME_SIZE:
        raise ValueError("OVERLAP must be less than FRAME_SIZE.")
    f, t, Zxx = signal.stft(audio, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    return f, t, Zxx

# --- ISTFT ---
def reconstruct_signal(Zxx, fs):
    _, x_rec = signal.istft(Zxx, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    return x_rec

# --- Improved Spectral Subtraction ---
def improved_spectral_subtraction(noisy, noise_ref, fs):
    _, _, Y = compute_spectrogram(noisy, fs)
    _, _, R = compute_spectrogram(noise_ref, fs)

    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    R_mag = np.abs(R)

    # --- Subtract max reference band-by-band ---
    S_hat_mag = np.maximum(Y_mag - np.max(R_mag, axis=1, keepdims=True), 0)

    # --- Remove small peaks ---
    mean_peak = np.mean(S_hat_mag, axis=0)
    mask = S_hat_mag < THRESHOLD_PEAK_RATIO * mean_peak
    S_hat_mag[mask] = 0

    # --- Reconstruct with original phase ---
    S_hat = S_hat_mag * np.exp(1j * Y_phase)
    return reconstruct_signal(S_hat, fs)

# --- Save audio ---
def save_audio(filename, fs, audio):
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    wav.write(filename, fs, audio)

# --- SNR ---
def compute_snr(clean, enhanced):
    noise = clean - enhanced
    snr = 10 * np.log10(np.sum(clean ** 2) / (np.sum(noise ** 2) + 1e-8))
    return snr

# --- Main Runner ---
def main():
    fs, noisy = load_audio("noisy_mixture.wav")
    _, clean = load_audio("clean_speech.wav")
    _, noise = load_audio("ambient_noise.wav")

    print("Running ISS method...")
    clean_est = improved_spectral_subtraction(noisy, noise, fs)
    save_audio("iss_output.wav", fs, clean_est)
    print("Saved ISS output as 'iss_output.wav'")

    snr_val = compute_snr(clean[:len(clean_est)], clean_est)
    print(f"SNR (ISS): {snr_val: .2f} dB")

if __name__ == "__main__":
    main()