import sounddevice as sd
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = load_model('C:\\Users\\OMOLP091\\Documents\\GitHub\\SOUND_DIRECTION_PROJECT\\sound_direction\\music_sound_identification_model.h5')

# Define the provided classes
provided_classes = [
    'Classical-Music-Low', 'Classical-Music-Medium', 'Classical-Music-High',
    'SoloPiano-Music-Low', 'SoloPiano-Music-Medium', 'SoloPiano-Music-High',
    'VideoGame-Music-Low', 'VideoGame-Music-Medium', 'VideoGame-Music-High',
    'IndieInstrumentals-Music-Low', 'IndieInstrumentals-Music-Medium', 'IndieInstrumentals-Music-High',
    'Harp-Music-Low', 'Harp-Music-Medium', 'Harp-Music-High',
    'Guitar-music-Low', 'Guitar-music-Medium', 'Guitar-music-High',
    'NewAge-Music-Low', 'NewAge-Music-Medium', 'NewAge-Music-High',
    'Pop-Music-Low', 'Pop-Music-Medium', 'Pop-Music-High',
    'Jazz-Music-Low', 'Jazz-Music-Medium', 'Jazz-Music-High',
    'Ambient-Music-Low', 'Ambient-Music-Medium', 'Ambient-Music-High',
    'Sitar-Music-Low', 'Sitar-Music-Medium', 'Sitar-Music-High',
    'NatureSounds-Music-Low', 'NatureSounds-Music-Medium', 'NatureSounds-Music-High',
    'InstrumentalQawwali-Music-Low', 'InstrumentalQawwali-Music-Medium', 'InstrumentalQawwali-Music-High',
    'IndianTabla-Music-Low', 'IndianTabla-Music-Medium', 'IndianTabla-Music-High',
    'Ghatam-Music-Low', 'Ghatam-Music-Medium', 'Ghatam-Music-High',
    'Bluegrass-Music-Low', 'Bluegrass-Music-Medium', 'Bluegrass-Music-High',
    'BhajanInstrumentals-Music-Low', 'BhajanInstrumentals-Music-Medium', 'BhajanInstrumentals-Music-High',
    'AvantGardeJazz-Music-Low', 'AvantGardeJazz-Music-Medium', 'AvantGardeJazz-Music-High'
]

# Initialize the encoder
encoder = LabelEncoder()

# Fit and transform the provided classes
encoded_classes = encoder.fit_transform(provided_classes)

# Assign the encoded classes to the encoder
encoder.classes_ = encoded_classes

# Define a function to process audio from the microphone
def process_audio(indata, frames, time, status):
    global encoder  # Use the global encoder variable
    # Extract audio data
    audio_data = np.array(indata.flatten(), dtype='float32')

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio_data, sr=44100, n_mfcc=64)
    mfcc_features = np.mean(mfcc_features, axis=1)
    mfcc_features = mfcc_features.reshape(1, -1)  # Reshape for model input

    # Perform prediction
    prediction = model.predict(mfcc_features)

    # Get the predicted class
    predicted_class = np.argmax(prediction)

    # Decode the predicted class
    predicted_class_name = encoder.inverse_transform([predicted_class])[0]

    # Print the predicted class
    print(f'The predicted class is: {predicted_class_name}')
    # stop the stream after prediction
    sd.stop()

# Start recording audio from the microphone
print("Recording... Press Ctrl+C to stop.")
duration = 5  # record for 5 seconds
recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
sd.wait()
process_audio(recording, None, None, None)
