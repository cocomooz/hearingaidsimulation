# optional to record raw audio for iss v sss testing
import pyaudio
import wave

def record_audio(filename="output.wav", duration=5, rate=44100):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate,
                    input=True, frames_per_buffer=CHUNK)

    print(f"Recording {duration} seconds to '{filename}'...")
    frames = []

    for _ in range(int(rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == "__main__":
    record_audio(filename="noisy_input.wav", duration=5)
    #record_audio(filename="reference_noise.wav", duration=5)
