from pydub import AudioSegment
import os

def split_wav(input_file, output_folder, split_duration=3600):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Load the input audio file directly
        audio = AudioSegment.from_file(input_file)
    except Exception as e:
        print("Error loading input file directly:", e)
        print("Trying alternative loading methods...")
        # Try loading the input audio file using different formats and codecs
        try:
            audio = AudioSegment.from_file(input_file, format="wav")
        except Exception as e:
            print("Error loading with wav format:", e)
            try:
                audio = AudioSegment.from_file(input_file, format="mp3")
            except Exception as e:
                print("Error loading with mp3 format:", e)
                # Add more alternative loading methods if needed
                return  # If all attempts fail, exit the function

    # Calculate number of chunks
    num_chunks = len(audio) // split_duration + 1

    # Split the audio into chunks of specified duration
    for i in range(num_chunks):
        start_time = i * split_duration
        end_time = min((i + 1) * split_duration, len(audio))
        chunk = audio[start_time:end_time]

        # Save the chunk as a new wav file with PCM signed 16-bit little-endian codec
        output_file = os.path.join(output_folder, f"Highway_sounds_{i}.wav")
        chunk.export(output_file, format="wav", codec="pcm_s16le")

if __name__ == "__main__":
    input_file = "Highway_sounds.wav"
    output_folder = "OUTPUT-WAV"
    split_duration = 5000  # 5 seconds in milliseconds

    split_wav(input_file, output_folder, split_duration)
