import speech_recognition as sr

# Lists every microphone and its index. Set BRUCE_MIC_INDEX to the one you want.
print("Available microphones:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"Microphone {index}: {name}")
