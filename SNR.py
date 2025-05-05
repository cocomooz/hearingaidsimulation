import numpy as np
import scipy.io.wavfile as wav

def load_audio(path):
    rate, data = wav.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    return rate, data.astype(np.float32)

def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)

def match_length(a, b):
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]

def mix_snr(clean, noise, snr_db):
    clean, noise = match_length(clean, noise)
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)

    target_noise_power = clean_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / (noise_power + 1e-8))

    noise_scaled = noise * scaling_factor
    mixed = clean + noise_scaled
    return mixed, noise_scaled

def save_audio(path, rate, data):
    data = np.clip(data, -32768, 32767).astype(np.int16)
    wav.write(path, rate, data)

def main():
    # === Replace these with your actual file paths ===
    clean_path = "clean_speech.wav"
    noise_path = "ambient_noise.wav"
    snr_db = 5

    fs1, clean = load_audio(clean_path)
    fs2, noise = load_audio(noise_path)
    assert fs1 == fs2, "rates do not match"

    clean = normalize_audio(clean)
    noise = normalize_audio(noise)

    mixed, scaled_noise = mix_snr(clean, noise, snr_db)

    save_audio("clean.wav", fs1, clean * 32767)
    save_audio("noise.wav", fs1, scaled_noise * 32767)
    save_audio("noisy_mixture.wav", fs1, mixed * 32767)

    print("Saved synthetic test files:")
    print("- clean.wav")
    print("- noise.wav")
    print("- noisy_mixture.wav")

if __name__ == "__main__":
    main()
