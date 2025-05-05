import pyaudio
import time
import numpy as np
import tkinter as tk
import threading
import matplotlib.pyplot as plt
import wave
from scipy.signal import spectrogram
import scipy.io.wavfile as wav
import scipy.signal as signal

# ------------------------
# Constants
# ------------------------
CHUNK = 1024
RATE = 44100
CHANNELS = 1
FORMAT = pyaudio.paInt16
ALPHA = 0.95
FRAME_SIZE = 1024
OVERLAP = 512
THRESHOLD_PEAK_RATIO = 0.5
NUM_MICS = 2
MIC_DISTANCE = 0.05
SPEED_OF_SOUND = 343

INPUT_DEVICE_INDEX = 0 # iPhone Speaker
OUTPUT_DEVICE_INDEX = 2  #MacBook Pro Speakers

raw_audio = []
processed_audio = []

# ------------------------
def supercardioid_gain(theta):
    return 0.3 + 0.7 * np.cos(theta)

def apply_directionality(audio, theta=0.0):
    gain = supercardioid_gain(theta)
    return gain * audio

def compute_spectrogram(audio, fs):
    f, t, Zxx = signal.stft(audio, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    print(f"Computed spectrogram: {Zxx.shape}")  # Debugging: Print spectrogram shape
    return f, t, Zxx

def reconstruct_signal(Zxx, fs):
    _, x_rec = signal.istft(Zxx, fs, nperseg=FRAME_SIZE, noverlap=OVERLAP)
    print(f"Reconstructed signal of length: {len(x_rec)}")  # Debugging: Signal length after ISTFT
    return x_rec

def improved_spectral_subtraction(noisy, noise_ref, fs):
    _, _, Y = compute_spectrogram(noisy, fs)
    _, _, R = compute_spectrogram(noise_ref, fs)
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)
    R_mag = np.abs(R)
    S_hat_mag = np.maximum(Y_mag - np.max(R_mag, axis=1, keepdims=True), 0)
    mean_peak = np.mean(S_hat_mag, axis=0)
    mask = S_hat_mag < THRESHOLD_PEAK_RATIO * mean_peak
    S_hat_mag[mask] = 0
    S_hat = S_hat_mag * np.exp(1j * Y_phase)
    print("Applied spectral subtraction.")
    return reconstruct_signal(S_hat, fs)

def save_audio(filename, fs, audio):
    audio = np.clip(audio, -32768, 32767).astype(np.int16)
    wav.write(filename, fs, audio)
    print(f"Audio saved to {filename}")

def delay_signal(sig, delay, fs):
    shift = int(round(delay * fs))
    delayed_sig = np.pad(sig, (shift, 0), 'constant')[:len(sig)]
    print(f"Applied delay of {delay} seconds.")
    return delayed_sig

def simulate_mic_array(source, fs, angle_deg):
    angle_rad = np.radians(angle_deg)
    delay = (MIC_DISTANCE * np.sin(angle_rad)) / SPEED_OF_SOUND
    mic1 = source
    mic2 = delay_signal(source, delay, fs)
    print(f"Simulated mic array with angle {angle_deg} degrees.")
    return np.stack([mic1, mic2], axis=0)

def lcmv_beamformer(Rxx, C, f):
    try:
        Rxx_inv = np.linalg.inv(Rxx)
        weights = Rxx_inv @ C @ np.linalg.inv(C.conj().T @ Rxx_inv @ C) @ f
        print("LCMV Beamforming applied.")
        return weights
    except np.linalg.LinAlgError:
        print("LCMV Beamforming failed: using default weights.")
        return np.ones(C.shape, dtype=np.complex128) / C.shape[0]

class AudioProcessor:
    def __init__(self, lcmv_enabled_var, look_direction_var, progress_var):
        self.p = pyaudio.PyAudio()
        self.stream_in = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                     frames_per_buffer=CHUNK, input_device_index=INPUT_DEVICE_INDEX)
        self.stream_out = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                                      frames_per_buffer=CHUNK, output_device_index=OUTPUT_DEVICE_INDEX)
        self.noise_estimate_basic_ss = np.zeros(CHUNK)
        self.running = True
        self.reduction_factor = 0.5
        self.noise_reference = None
        self.learning_noise = False
        self.noise_buffer = []
        self.noise_buffer_length = 5
        self.noise_buffer_size = int(self.noise_buffer_length * RATE / CHUNK)
        self.lcmv_enabled = lcmv_enabled_var
        self.look_direction = look_direction_var
        self.progress_var = progress_var
        self.total_processing_time = 0.0
        self.total_audio_duration = 0.0

    def update_factor(self, val):
        self.reduction_factor = float(val)
        print(f"Noise reduction factor updated to {self.reduction_factor}")

    def stop(self):
        print("Stopping AudioProcessor...")
        self.running = False
        try:
            if self.stream_in.is_active():
                self.stream_in.stop_stream()
            self.stream_in.close()
            if self.stream_out.is_active():
                self.stream_out.stop_stream()
            self.stream_out.close()
            self.p.terminate()
            print("Audio streams and PyAudio terminated.")
        except Exception as e:
            print("Error while stopping AudioProcessor:", e)

    def start_noise_learning(self):
        print("Starting noise learning...")
        self.learning_noise = True
        self.noise_buffer = []
        self.progress_var.set(0)

    def apply_lcmv(self, mic_signals, look_angle_deg=0.0):
        n_sensors = mic_signals.shape[0]
        if n_sensors < 2:
            return mic_signals[0] if n_sensors == 1 else np.zeros(mic_signals.shape[1])
        Rxx = np.cov(mic_signals)
        look_angle_rad = np.radians(look_angle_deg)
        steering_vector = np.exp(-1j * 2 * np.pi * RATE / SPEED_OF_SOUND * MIC_DISTANCE * np.sin(look_angle_rad) * np.arange(n_sensors))
        steering_vector = steering_vector.reshape(-1, 1)
        C = steering_vector
        f = np.array([[1.0]])
        weights = lcmv_beamformer(Rxx, C, f)
        _, _, Zxx = compute_spectrogram(mic_signals[0, :], RATE)
        beamformed_spectrum = np.zeros_like(Zxx, dtype=np.complex128)
        for i in range(Zxx.shape[1]):
            frame_data_fft = np.fft.fft(mic_signals[:, i])
            beamformed_spectrum[:, i] = np.dot(weights.conj().T, frame_data_fft.reshape(-1, 1)).flatten()
        print("LCMV beamforming applied to the audio.")
        return reconstruct_signal(beamformed_spectrum, RATE)

    def process_audio(self):
        while self.running:
            start_time = time.time()
            data = self.stream_in.read(CHUNK, exception_on_overflow=False)
            audio_array = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            raw_audio.append(audio_array.copy())
            print("Captured audio chunk of size:", len(audio_array))
            try:
                simulated_mics = simulate_mic_array(audio_array.copy(), RATE, self.look_direction.get())
                print(f"Simulated mic array with angle {self.look_direction.get()} degrees.")
            except Exception as e:
                print("Error in simulate_mic_array:", e)
                continue

            # LCMV BEAMFORMING
            if self.lcmv_enabled.get():
                try:
                    beamformed_audio = self.apply_lcmv(simulated_mics, self.look_direction.get())
                    print("LCMV beamforming applied to the audio.")
                except Exception as e:
                    print("LCMV beamforming failed:", e)
                    beamformed_audio = simulated_mics[0]
            else:
                beamformed_audio = audio_array

            audio_to_process = beamformed_audio[:CHUNK]

            # NOISE LEARNING
            if self.learning_noise:
                self.noise_buffer.append(audio_to_process)
                progress = int(100 * len(self.noise_buffer) / self.noise_buffer_size)
                self.progress_var.set(min(progress, 100))
                print(f"Learning noise... buffer size: {len(self.noise_buffer)}/{self.noise_buffer_size}")
                if len(self.noise_buffer) >= self.noise_buffer_size:
                    print("Noise learning complete.")
                    noise_segment = np.concatenate(self.noise_buffer, axis=0)
                    _, _, noise_spectrogram = compute_spectrogram(noise_segment, RATE)
                    self.noise_reference = np.abs(noise_spectrogram)
                    self.learning_noise = False
                    self.noise_buffer = []
                continue

            # BACKUP TO SSS IF FAILURE
            if self.noise_reference is not None:
                try:
                    processed_audio_np = improved_spectral_subtraction(audio_to_process,
                                                                       np.mean(self.noise_reference, axis=1,
                                                                               keepdims=True), RATE)
                    print("Spectral subtraction applied.")
                except Exception as e:
                    print("Error in spectral subtraction:", e)
                    processed_audio_np = audio_to_process
            else:
                spectrum = np.fft.fft(audio_to_process)
                magnitude = np.abs(spectrum)
                phase = np.angle(spectrum)
                self.noise_estimate_basic_ss = ALPHA * self.noise_estimate_basic_ss + (1 - ALPHA) * magnitude
                subtracted = np.maximum(magnitude - self.reduction_factor * self.noise_estimate_basic_ss, 0)
                processed_spectrum = subtracted * np.exp(1j * phase)
                processed_audio_np = np.fft.ifft(processed_spectrum).real
                print("Basic spectral subtraction applied.")

            processed_audio_np *= 5.0

            print("Processed audio mean:", np.mean(processed_audio_np))
            print("Processed audio max:", np.max(processed_audio_np))

            processed_audio.append(processed_audio_np.copy())
            processed_audio_np = np.clip(processed_audio_np, -32768, 32767).astype(np.int16)
            self.stream_out.write(processed_audio_np.tobytes())

            end_time = time.time()

            self.total_processing_time += (end_time - start_time)
            self.total_audio_duration += CHUNK / RATE

def save_wav(filename, frames, rate=RATE):
    flat = np.concatenate(frames).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(flat.tobytes())
    print(f"Saved WAV file: {filename}")

def plot_spec(audio, title, filename):
    f, t, Sxx = spectrogram(audio, fs=RATE)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.colorbar(label='Power')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Spectrogram saved as {filename}")

def start_ui(audio_processor_class):
    root = tk.Tk()
    root.title("Noise Reduction Controller")

    lcmv_enabled_var = tk.BooleanVar(value=False)
    look_direction_var = tk.DoubleVar()
    look_direction_var.set(0.0)
    progress_var = tk.IntVar(value=0)
    processor = audio_processor_class(lcmv_enabled_var, look_direction_var, progress_var)

    lcmv_checkbutton = tk.Checkbutton(root, text="Enable LCMV", variable=lcmv_enabled_var)
    lcmv_checkbutton.pack()

    tk.Label(root, text="LCMV Look Direction (degrees):").pack()
    tk.Entry(root, textvariable=look_direction_var).pack()

    tk.Label(root, text="Noise Reduction Level (Basic SS)").pack()
    noise_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.01,
                            orient=tk.HORIZONTAL, command=processor.update_factor)
    noise_slider.set(processor.reduction_factor)
    noise_slider.pack()

    learn_noise_button = tk.Button(root, text="Learn Noise", command=processor.start_noise_learning)
    learn_noise_button.pack()

    progress_bar = tk.Scale(root, from_=0, to=100, orient=tk.HORIZONTAL, length=200,
                            label="Noise Learning Progress", variable=progress_var)
    progress_bar.pack()

    def on_closing():
            processor.stop()
            root.destroy()
            print("Saving WAV files and spectrograms...")
            save_wav("raw_input.wav", raw_audio)
            save_wav("processed_output.wav", processed_audio)
            plot_spec(np.concatenate(raw_audio), "Raw Input", "raw_spectrogram.png")
            plot_spec(np.concatenate(processed_audio), "Processed Output", "processed_spectrogram.png")

            # --- RTF reporting ---
            if processor.total_audio_duration > 0:
                rtf = processor.total_processing_time / processor.total_audio_duration
                print(f"\n Real-Time Factor (RTF): {rtf:.4f}")
            else:
                print("RTF could not be calculated (no audio processed).")

    root.protocol("WM_DELETE_WINDOW", on_closing)
    threading.Thread(target=processor.process_audio, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    start_ui(AudioProcessor)
