import pyaudio
import wave
import os
import whisper

# Record a chunk of audio and save it to a file
def record_chunk(p, stream, file_path, chunk_length=1):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):  # 16000 Hz, 1-second chunks
        data = stream.read(1024)
        frames.append(data)
    
    # Save frames to a .wav file
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

# Main function
def main():
    # Initialize Whisper model
    model = whisper.load_model("base")

    # Set up PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )

    accumulated_transcription = ""  # To store all transcriptions

    try:
        print("Recording and transcribing... Press Ctrl+C to stop.")
        while True:
            chunk_file = "temp_chunk.wav"
            record_chunk(p, stream, chunk_file)

            # Transcribe the chunk
            result = model.transcribe(chunk_file, fp16=False)
            transcription = result["text"].strip()
            print(f"Transcription: {transcription}")

            # Add to accumulated transcription
            accumulated_transcription += transcription + " "

            # Clean up temporary file
            os.remove(chunk_file)
    except KeyboardInterrupt:
        print("\nStopping...")
        # Save accumulated transcription to a file
        with open("transcription_log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)
        print("Transcription saved to transcription_log.txt.")
    finally:
        # Clean up PyAudio
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
