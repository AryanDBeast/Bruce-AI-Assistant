import speech_recognition as sr

# List all available microphone devices
print("Available microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")
