# Voice Bot Pipeline Explanation

## 🔄 Complete Voice Bot Pipeline:

### 1. 🎙️ **Your Voice → STT (Speech-to-Text)**
```
You speak: "What's my account balance?"
  ↓
Whisper STT: "What's my account balance?"
```

### 2. 🧠 **Text → LLM (AI Processing)**  
```
Input: "What's my account balance?"
  ↓
LLM: "Your current account balance is $5,432.10"
```

### 3. 🔊 **Text → TTS (Text-to-Speech)**
```
Text: "Your current account balance is $5,432.10"
  ↓
TTS Engine: Spoken audio output
```

## 📚 Library Breakdown:

### STT (Speech-to-Text) Options:
- **OpenAI Whisper**: `openai-whisper` - High quality, local
- **Google Speech**: `SpeechRecognition` - Free, cloud-based
- **Azure Speech**: `azure-cognitiveservices-speech` - Enterprise
- **AWS Transcribe**: `boto3` - Cloud service

### TTS (Text-to-Speech) Options:
- **pyttsx3**: Local, works offline, multiple voices
- **gTTS (Google)**: Cloud-based, natural voices
- **Azure Speech**: Enterprise-grade, many languages
- **AWS Polly**: Cloud service with neural voices
- **Kokoro TTS**: Advanced neural TTS (mentioned in Sprint 6)

## 🎯 Why The Confusion?

People often think:
❌ "Whisper does TTS" 
✅ "Whisper does STT, need separate TTS"

## 🚀 Recommended Setup for Voice Test:

### For Speech Recognition (STT):
```bash
# Option 1: Google Speech (easier setup)
pip install SpeechRecognition

# Option 2: OpenAI Whisper (better quality)
pip install openai-whisper
```

### For Text-to-Speech (TTS):
```bash
# Option 1: Local TTS (works offline)
pip install pyttsx3

# Option 2: Google TTS (cloud, better quality)
pip install gTTS

# Option 3: Azure Speech (enterprise)
pip install azure-cognitiveservices-speech
```

### For Audio Processing:
```bash
pip install pyaudio  # For microphone input
pip install playsound  # For audio playback
```

## ⚠️ Common Issues:

### PyAudio Installation (Windows):
```bash
# If pyaudio fails:
pip install pipwin
pipwin install pyaudio
```

### Whisper vs TTS Libraries:
- ❌ Don't use Whisper for TTS
- ✅ Use Whisper for STT + separate TTS library
- ✅ Use pyttsx3 for simple local TTS
- ✅ Use gTTS for better quality cloud TTS
