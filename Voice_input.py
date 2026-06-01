import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

class VoiceInput:
    def __init__(self):
        self.model = None

    def load_model(self): #Loading the model that is used to transcibe whatever is told in speech(Audio input) to text
        self.model = WhisperModel(
            "base",
            device="cpu",
            compute_type="int8"
        )
        print("Whisper model loaded.") #Informing that the Whisper model has been loaded

    def record_audio(
        self,
        duration=5, #5 seconds of audio recorded
        sample_rate=16000 #Number of samples from the audio wave taken per second (Number of snapshots from the sound wave each second)
    ): #Function to record the voice input so that it can be saved and later the words from the voice input can be extracted
        print("\nSpeak now...\n")
        #This statement records the audio input which we give
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1
        )
        sd.wait() #The code will wait till the audio is recorded completely and then only writes the recording into query.wav. If we don't do this, the recording might stop to take inputs abruptly before we complete recording the co

        write(
            "query.wav",
            sample_rate,
            audio
        )
        print("Recording complete.")

    def transcribe_audio(self):
        segments, info = self.model.transcribe(
            "query.wav"
        )

        query = ""
        for segment in segments:
            query += segment.text
        return query.strip()
    
if __name__ == "__main__": #testing whether the voice input is actually taken and transcribed
    voice = VoiceInput()
    voice.load_model()
    voice.record_audio()
    query = voice.transcribe_audio()
    print("\nDetected Query:\n")
    print(query)