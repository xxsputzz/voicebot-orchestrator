#!/usr/bin/env python3

import sys
import os

# Add the voicebot_orchestrator to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'voicebot_orchestrator'))

from enhanced_tts_manager import EnhancedTTSManager
from flask import Flask, request, jsonify, Response
import io

app = Flask(__name__)

# Initialize TTS manager
print("🔄 Starting TTS service with Unicode fix...")
tts_manager = EnhancedTTSManager()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "Enhanced TTS"})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    try:
        data = request.json
        text = data.get('text', '')
        engine = data.get('engine', 'KOKORO')
        voice = data.get('voice', 'default')
        
        print(f"🎤 TTS Request: '{text}' using {engine}")
        
        # Generate speech using our enhanced manager with Unicode fix
        audio_data = tts_manager.generate_speech(text, engine, voice)
        
        if audio_data:
            print(f"✅ TTS Success: Generated {len(audio_data)} bytes")
            return Response(audio_data, mimetype='audio/wav')
        else:
            return jsonify({"error": "Failed to generate audio"}), 500
            
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Enhanced TTS Service starting on port 8004...")
    print("🛠️  Includes Unicode text sanitization fix")
    app.run(host='0.0.0.0', port=8004, debug=False)
