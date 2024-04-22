import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from tensorflow.keras.models import load_model
import pyttsx3

# Constants
SOUND_SPEED = 343  # Speed of sound in m/s
MIC_DISTANCE = 0.1  # Distance between the microphones in meters
MIC_OFFSET = 0.03  # Offset of the microphones in meters
LABELS = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling',
          'gun_shot', 'jackhammer', 'siren', 'street_music']  # Sound class labels

# Load pre-trained sound classification model
model = load_model('C:\\Users\\OMOLP091\\Documents\\GitHub\\sound-repo\\sound_identification_model.h5')  # Replace 'sound_classification_model.h5' with your model path

# Initialize the speech synthesizer
engine = pyttsx3.init()

# Function to calculate the angle of arrival based on time delay
def calc_angle(tdoa):
    return np.arctan(tdoa * SOUND_SPEED / MIC_DISTANCE) * 180 / np.pi

# Function to capture audio from the microphones
def capture_audio(duration):
    fs = 44100  # Sample rate
    channels = 1  # Mono audio

    # Record audio from the first microphone
    print("Recording from mic 1...")
    mic1_audio = sd.rec(int(fs * duration), samplerate=fs, channels=channels, dtype='float32')
    sd.wait()

    # Record audio from the second microphone
    print("Recording from mic 2...")
    mic2_audio = sd.rec(int(fs * duration), samplerate=fs, channels=channels, dtype='float32')
    sd.wait()

    return mic1_audio[:, 0], mic2_audio[:, 0]

# Function to localize sound direction and identify the object
def localize_sound_direction_and_identify_object(duration):
    mic1_audio, mic2_audio = capture_audio(duration)

    # Combine audio from both microphones
    combined_audio = np.hstack((mic1_audio, mic2_audio))

    # Split the combined audio into segments of size 64
    segment_size = 64
    num_segments = len(combined_audio) // segment_size
    segments = np.array_split(combined_audio[:num_segments * segment_size], num_segments)

    # Classify each segment and store the predictions
    predictions = []
    for segment in segments:
        sound_features = np.reshape(segment, (1, -1))
        prediction = model.predict(sound_features)
        predictions.append(prediction)

    # Average the predictions for all segments
    average_prediction = np.mean(predictions, axis=0)
    predicted_class = LABELS[np.argmax(average_prediction)]

    # Cross-correlation to find the time delay
    corr = np.correlate(mic1_audio, mic2_audio, mode='full')
    delay_sample = np.argmax(corr) - len(mic1_audio) + 1
    delay_sec = delay_sample / 44100

    # Calculate the angle of arrival
    tdoa = delay_sec + MIC_OFFSET / SOUND_SPEED
    angle = calc_angle(tdoa)

    if angle < -45:
        direction = "Left"
    elif angle > 45:
        direction = "Right"
    elif -45 <= angle <= 45:
        direction = "Forward"
    else:
        direction = "Backward"

    # Speak the direction and identified object
    engine.say(f"The sound is coming from: {direction} and it is identified as: {predicted_class}")
    engine.runAndWait()

    return direction, predicted_class

# Main function
if __name__ == "__main__":
    duration = 0.5  # Duration to record audio in seconds
    direction, sound_class = localize_sound_direction_and_identify_object(duration)
    print(f"The sound is coming from: {direction} and it is identified as: {sound_class}")
