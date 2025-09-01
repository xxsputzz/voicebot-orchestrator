"""
Interactive TTS User Interface
Web-based interface for text input, voice selection, and streaming synthesis
"""
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import uuid
import io
import time
from datetime import datetime

from voicebot_orchestrator.streaming_tts import StreamingTTSEngine, StreamingConfig, create_streaming_engine

# Data models
class SynthesisRequest(BaseModel):
    text: str
    voice: str = "default"
    emotion: str = "neutral"
    speaking_style: str = "normal"
    speed: float = 1.0
    pitch: float = 1.0
    output_format: str = "wav"
    streaming: bool = True
    chunk_size: int = 200

class VoicePreviewRequest(BaseModel):
    voice: str
    emotion: str = "neutral"
    text: str = "Hello, this is a preview of my voice."

class BatchSynthesisRequest(BaseModel):
    texts: List[str]
    voice: str = "default"
    emotion: str = "neutral"
    speaking_style: str = "normal"
    speed: float = 1.0
    pitch: float = 1.0

# FastAPI app
app = FastAPI(title="Interactive TTS Interface", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
streaming_engines: Dict[str, StreamingTTSEngine] = {}
active_syntheses: Dict[str, Dict] = {}
user_preferences: Dict[str, Dict] = {}

@app.on_event("startup")
async def startup():
    """Initialize streaming TTS engine"""
    global streaming_engines
    
    # Create default streaming engine
    config = StreamingConfig(
        chunk_size=200,
        max_concurrent=3,
        buffer_size=5
    )
    
    engine = await create_streaming_engine(config=config)
    streaming_engines["default"] = engine
    
    print("[OK] Interactive TTS Interface initialized")

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the interactive TTS interface"""
    return HTMLResponse(content=get_html_interface(), status_code=200)

@app.get("/api/voices")
async def get_available_voices():
    """Get available voices with categories and samples"""
    engine = streaming_engines["default"]
    options = engine.tts_engine.get_available_options()
    
    # Add preview samples for each voice
    voice_catalog = {}
    for category, voices in options["voices"].items():
        voice_catalog[category] = {}
        for voice_name, voice_info in voices.items():
            voice_catalog[category][voice_name] = {
                **voice_info,
                "preview_url": f"/api/preview/{voice_name}",
                "sample_text": get_sample_text_for_voice(voice_name)
            }
    
    return {
        "voices": voice_catalog,
        "emotions": options["emotions"],
        "speaking_styles": options["speaking_styles"],
        "total_voices": sum(len(voices) for voices in options["voices"].values())
    }

@app.post("/api/preview/{voice}")
async def generate_voice_preview(voice: str, request: VoicePreviewRequest):
    """Generate a quick voice preview"""
    engine = streaming_engines["default"]
    
    try:
        audio_bytes = await engine.tts_engine.synthesize_speech(
            text=request.text,
            voice=voice,
            emotion=request.emotion,
            high_quality=False  # Fast preview
        )
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=preview_{voice}.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")

@app.post("/api/synthesize")
async def synthesize_text(request: SynthesisRequest):
    """Start text synthesis (streaming or standard)"""
    synthesis_id = str(uuid.uuid4())
    
    if request.streaming and len(request.text) > 500:
        # Use streaming for long texts
        return await start_streaming_synthesis(synthesis_id, request)
    else:
        # Use standard synthesis for short texts
        return await standard_synthesis(synthesis_id, request)

async def start_streaming_synthesis(synthesis_id: str, request: SynthesisRequest):
    """Start streaming synthesis and return status"""
    engine = streaming_engines["default"]
    
    # Store synthesis info
    active_syntheses[synthesis_id] = {
        "id": synthesis_id,
        "text": request.text,
        "voice": request.voice,
        "emotion": request.emotion,
        "status": "starting",
        "start_time": datetime.now(),
        "chunks_completed": 0,
        "total_chunks": 0,
        "audio_urls": []
    }
    
    # Start background task
    asyncio.create_task(process_streaming_synthesis(synthesis_id, request, engine))
    
    return {
        "synthesis_id": synthesis_id,
        "status": "started",
        "streaming": True,
        "websocket_url": f"/ws/synthesis/{synthesis_id}",
        "status_url": f"/api/synthesis/{synthesis_id}/status"
    }

async def standard_synthesis(synthesis_id: str, request: SynthesisRequest):
    """Standard non-streaming synthesis"""
    engine = streaming_engines["default"]
    
    try:
        audio_bytes = await engine.tts_engine.synthesize_speech(
            text=request.text,
            voice=request.voice,
            emotion=request.emotion,
            speaking_style=request.speaking_style,
            speed=request.speed,
            pitch=request.pitch
        )
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=synthesis_{synthesis_id}.wav",
                "X-Synthesis-ID": synthesis_id
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

async def process_streaming_synthesis(synthesis_id: str, request: SynthesisRequest, engine: StreamingTTSEngine):
    """Process streaming synthesis in background"""
    try:
        audio_chunks = []
        
        async for chunk, audio_bytes in engine.stream_synthesis(
            text=request.text,
            stream_id=synthesis_id,
            voice=request.voice,
            emotion=request.emotion,
            speaking_style=request.speaking_style,
            speed=request.speed,
            pitch=request.pitch
        ):
            # Store audio chunk
            chunk_filename = f"chunk_{synthesis_id}_{chunk.id}.wav"
            audio_chunks.append({
                "chunk_id": chunk.id,
                "filename": chunk_filename,
                "size": len(audio_bytes),
                "duration_ms": chunk.duration_ms
            })
            
            # Update synthesis status
            if synthesis_id in active_syntheses:
                active_syntheses[synthesis_id]["chunks_completed"] = chunk.id + 1
                active_syntheses[synthesis_id]["audio_urls"] = audio_chunks
        
        # Mark as completed
        if synthesis_id in active_syntheses:
            active_syntheses[synthesis_id]["status"] = "completed"
            active_syntheses[synthesis_id]["completed_time"] = datetime.now()
            
    except Exception as e:
        if synthesis_id in active_syntheses:
            active_syntheses[synthesis_id]["status"] = "error"
            active_syntheses[synthesis_id]["error"] = str(e)

@app.get("/api/synthesis/{synthesis_id}/status")
async def get_synthesis_status(synthesis_id: str):
    """Get status of a synthesis job"""
    if synthesis_id not in active_syntheses:
        raise HTTPException(status_code=404, detail="Synthesis not found")
    
    status = active_syntheses[synthesis_id].copy()
    
    # Add progress calculation
    if status["total_chunks"] > 0:
        status["progress_percent"] = (status["chunks_completed"] / status["total_chunks"]) * 100
    else:
        status["progress_percent"] = 0
    
    return status

@app.post("/api/batch")
async def batch_synthesis(request: BatchSynthesisRequest):
    """Process multiple texts in batch"""
    batch_id = str(uuid.uuid4())
    results = []
    
    for i, text in enumerate(request.texts):
        try:
            synthesis_request = SynthesisRequest(
                text=text,
                voice=request.voice,
                emotion=request.emotion,
                speaking_style=request.speaking_style,
                speed=request.speed,
                pitch=request.pitch,
                streaming=False
            )
            
            result = await standard_synthesis(f"{batch_id}_{i}", synthesis_request)
            results.append({
                "index": i,
                "status": "completed",
                "text_preview": text[:50] + "..." if len(text) > 50 else text
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e),
                "text_preview": text[:50] + "..." if len(text) > 50 else text
            })
    
    return {
        "batch_id": batch_id,
        "total_items": len(request.texts),
        "results": results
    }

@app.post("/api/upload")
async def upload_text_file(file: UploadFile = File(...)):
    """Upload text file for synthesis"""
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only .txt and .md files supported")
    
    content = await file.read()
    text = content.decode('utf-8')
    
    # Estimate processing time
    engine = streaming_engines["default"]
    estimate = await engine.estimate_processing_time(text)
    
    return {
        "filename": file.filename,
        "text_length": len(text),
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "estimated_processing": estimate,
        "recommended_streaming": len(text) > 1000
    }

@app.websocket("/ws/synthesis/{synthesis_id}")
async def websocket_synthesis_updates(websocket: WebSocket, synthesis_id: str):
    """WebSocket for real-time synthesis updates"""
    await websocket.accept()
    
    try:
        while True:
            if synthesis_id in active_syntheses:
                status = active_syntheses[synthesis_id]
                await websocket.send_json(status)
                
                if status["status"] in ["completed", "error"]:
                    break
            
            await asyncio.sleep(1)  # Update every second
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

@app.get("/api/statistics")
async def get_engine_statistics():
    """Get TTS engine statistics"""
    engine = streaming_engines["default"]
    stats = engine.get_statistics()
    
    stats.update({
        "active_syntheses": len(active_syntheses),
        "total_users": len(user_preferences),
        "uptime": time.time() - startup_time if 'startup_time' in globals() else 0
    })
    
    return stats

# Helper functions
def get_sample_text_for_voice(voice_name: str) -> str:
    """Get appropriate sample text for a voice"""
    samples = {
        "sophia": "Hello! I'm Sophia, your friendly AI assistant.",
        "aria": "Welcome to our presentation. Let me guide you through the content.",
        "professional": "Good morning. This is a professional announcement.",
        "conversational": "Hey there! How's your day going?",
        "narrative": "Once upon a time, in a distant land...",
        "dramatic": "This is the most important announcement you'll hear today!"
    }
    
    return samples.get(voice_name, "This is a sample of my voice. How do I sound?")

def get_html_interface() -> str:
    """Generate the HTML interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive TTS Interface</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }
            .header { text-align: center; margin-bottom: 30px; }
            .input-section { margin-bottom: 20px; }
            .voice-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
            .voice-card { border: 1px solid #ddd; padding: 15px; border-radius: 8px; cursor: pointer; }
            .voice-card:hover { background: #f0f0f0; }
            .voice-card.selected { border-color: #007bff; background: #e3f2fd; }
            .controls { display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }
            .control-group { display: flex; flex-direction: column; }
            .control-group label { font-weight: bold; margin-bottom: 5px; }
            .control-group select, .control-group input { padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
            .text-input { width: 100%; min-height: 200px; padding: 15px; border: 1px solid #ddd; border-radius: 8px; font-size: 14px; }
            .buttons { text-align: center; margin: 20px 0; }
            .btn { padding: 12px 24px; margin: 0 10px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
            .btn-primary { background: #007bff; color: white; }
            .btn-secondary { background: #6c757d; color: white; }
            .progress { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; margin: 20px 0; }
            .progress-bar { height: 100%; background: #007bff; border-radius: 10px; transition: width 0.3s; }
            .status { padding: 15px; margin: 20px 0; border-radius: 8px; }
            .status.success { background: #d4edda; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; border: 1px solid #f5c6cb; }
            .status.info { background: #d1ecf1; border: 1px solid #bee5eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Interactive TTS Studio</h1>
                <p>Advanced text-to-speech with streaming synthesis and voice options</p>
            </div>
            
            <div class="input-section">
                <h3>Text Input</h3>
                <textarea id="textInput" class="text-input" placeholder="Enter your text here... (supports long texts with streaming synthesis)">Welcome to the Interactive TTS Studio! This advanced text-to-speech system supports multiple voices, emotions, speaking styles, and streaming synthesis for long texts. Try selecting different voices and emotions to hear how they change the speech output.</textarea>
            </div>
            
            <div class="input-section">
                <h3>Voice Selection</h3>
                <div id="voiceGrid" class="voice-grid">
                    <!-- Voices will be loaded here -->
                </div>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label>Emotion</label>
                    <select id="emotionSelect">
                        <option value="neutral">Neutral</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Speaking Style</label>
                    <select id="styleSelect">
                        <option value="normal">Normal</option>
                    </select>
                </div>
                <div class="control-group">
                    <label>Speed</label>
                    <input type="range" id="speedSlider" min="0.5" max="2.0" step="0.1" value="1.0">
                    <span id="speedValue">1.0x</span>
                </div>
                <div class="control-group">
                    <label>Pitch</label>
                    <input type="range" id="pitchSlider" min="0.5" max="2.0" step="0.1" value="1.0">
                    <span id="pitchValue">1.0x</span>
                </div>
            </div>
            
            <div class="buttons">
                <button id="previewBtn" class="btn btn-secondary">🔊 Preview Voice</button>
                <button id="synthesizeBtn" class="btn btn-primary">🎵 Synthesize Audio</button>
                <button id="streamBtn" class="btn btn-primary">📡 Stream Synthesis</button>
            </div>
            
            <div id="progressSection" style="display: none;">
                <h3>Synthesis Progress</h3>
                <div class="progress">
                    <div id="progressBar" class="progress-bar" style="width: 0%;"></div>
                </div>
                <div id="progressText">Preparing synthesis...</div>
            </div>
            
            <div id="statusSection"></div>
            
            <div id="resultsSection" style="display: none;">
                <h3>Generated Audio</h3>
                <audio id="audioPlayer" controls style="width: 100%; margin: 10px 0;"></audio>
                <div id="downloadLinks"></div>
            </div>
        </div>
        
        <script>
            // JavaScript for interactive functionality will be loaded here
            let selectedVoice = 'sophia';
            let voices = {};
            let emotions = {};
            let styles = {};
            
            // Load available voices and options
            async function loadVoices() {
                try {
                    const response = await fetch('/api/voices');
                    const data = await response.json();
                    voices = data.voices;
                    emotions = data.emotions;
                    styles = data.speaking_styles;
                    
                    renderVoiceGrid();
                    populateSelects();
                } catch (error) {
                    showStatus('Error loading voices: ' + error.message, 'error');
                }
            }
            
            function renderVoiceGrid() {
                const grid = document.getElementById('voiceGrid');
                grid.innerHTML = '';
                
                Object.entries(voices).forEach(([category, categoryVoices]) => {
                    Object.entries(categoryVoices).forEach(([voiceName, voiceInfo]) => {
                        const card = document.createElement('div');
                        card.className = 'voice-card';
                        card.onclick = () => selectVoice(voiceName);
                        
                        card.innerHTML = `
                            <h4>${voiceName}</h4>
                            <p><strong>Gender:</strong> ${voiceInfo.gender || 'Unknown'}</p>
                            <p><strong>Style:</strong> ${voiceInfo.style || 'Standard'}</p>
                            <p><strong>Accent:</strong> ${voiceInfo.accent || 'Neutral'}</p>
                        `;
                        
                        grid.appendChild(card);
                    });
                });
                
                // Select first voice by default
                if (Object.keys(voices).length > 0) {
                    const firstCategory = Object.keys(voices)[0];
                    const firstVoice = Object.keys(voices[firstCategory])[0];
                    selectVoice(firstVoice);
                }
            }
            
            function selectVoice(voiceName) {
                selectedVoice = voiceName;
                
                // Update UI
                document.querySelectorAll('.voice-card').forEach(card => {
                    card.classList.remove('selected');
                });
                
                event.target.closest('.voice-card').classList.add('selected');
            }
            
            function populateSelects() {
                // Populate emotions
                const emotionSelect = document.getElementById('emotionSelect');
                emotionSelect.innerHTML = '';
                
                Object.entries(emotions).forEach(([category, categoryEmotions]) => {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = category.charAt(0).toUpperCase() + category.slice(1);
                    
                    categoryEmotions.forEach(emotion => {
                        const option = document.createElement('option');
                        option.value = emotion;
                        option.textContent = emotion.replace('_', ' ');
                        optgroup.appendChild(option);
                    });
                    
                    emotionSelect.appendChild(optgroup);
                });
                
                // Populate speaking styles
                const styleSelect = document.getElementById('styleSelect');
                styleSelect.innerHTML = '';
                
                styles.forEach(style => {
                    const option = document.createElement('option');
                    option.value = style;
                    option.textContent = style.charAt(0).toUpperCase() + style.slice(1);
                    styleSelect.appendChild(option);
                });
            }
            
            // Event listeners
            document.getElementById('speedSlider').oninput = function(e) {
                document.getElementById('speedValue').textContent = e.target.value + 'x';
            };
            
            document.getElementById('pitchSlider').oninput = function(e) {
                document.getElementById('pitchValue').textContent = e.target.value + 'x';
            };
            
            document.getElementById('previewBtn').onclick = generatePreview;
            document.getElementById('synthesizeBtn').onclick = synthesizeText;
            document.getElementById('streamBtn').onclick = streamSynthesis;
            
            async function generatePreview() {
                const text = "This is a preview of my voice with the current settings.";
                const emotion = document.getElementById('emotionSelect').value;
                
                try {
                    const response = await fetch(`/api/preview/${selectedVoice}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ voice: selectedVoice, emotion: emotion, text: text })
                    });
                    
                    if (response.ok) {
                        const audioBlob = await response.blob();
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = document.getElementById('audioPlayer');
                        audio.src = audioUrl;
                        document.getElementById('resultsSection').style.display = 'block';
                        showStatus('Voice preview generated successfully!', 'success');
                    } else {
                        throw new Error('Preview generation failed');
                    }
                } catch (error) {
                    showStatus('Error generating preview: ' + error.message, 'error');
                }
            }
            
            async function synthesizeText() {
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    showStatus('Please enter some text to synthesize.', 'error');
                    return;
                }
                
                const request = {
                    text: text,
                    voice: selectedVoice,
                    emotion: document.getElementById('emotionSelect').value,
                    speaking_style: document.getElementById('styleSelect').value,
                    speed: parseFloat(document.getElementById('speedSlider').value),
                    pitch: parseFloat(document.getElementById('pitchSlider').value),
                    streaming: false
                };
                
                try {
                    showProgress('Synthesizing audio...', 0);
                    
                    const response = await fetch('/api/synthesize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(request)
                    });
                    
                    if (response.ok) {
                        const audioBlob = await response.blob();
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = document.getElementById('audioPlayer');
                        audio.src = audioUrl;
                        
                        document.getElementById('resultsSection').style.display = 'block';
                        hideProgress();
                        showStatus('Audio synthesis completed successfully!', 'success');
                    } else {
                        throw new Error('Synthesis failed');
                    }
                } catch (error) {
                    hideProgress();
                    showStatus('Error during synthesis: ' + error.message, 'error');
                }
            }
            
            async function streamSynthesis() {
                const text = document.getElementById('textInput').value;
                if (!text.trim()) {
                    showStatus('Please enter some text to synthesize.', 'error');
                    return;
                }
                
                const request = {
                    text: text,
                    voice: selectedVoice,
                    emotion: document.getElementById('emotionSelect').value,
                    speaking_style: document.getElementById('styleSelect').value,
                    speed: parseFloat(document.getElementById('speedSlider').value),
                    pitch: parseFloat(document.getElementById('pitchSlider').value),
                    streaming: true
                };
                
                try {
                    showProgress('Starting streaming synthesis...', 0);
                    
                    const response = await fetch('/api/synthesize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(request)
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        monitorStreamingSynthesis(result.synthesis_id);
                    } else {
                        throw new Error('Failed to start streaming synthesis');
                    }
                } catch (error) {
                    hideProgress();
                    showStatus('Error starting synthesis: ' + error.message, 'error');
                }
            }
            
            function monitorStreamingSynthesis(synthesisId) {
                const ws = new WebSocket(`ws://localhost:8015/ws/synthesis/${synthesisId}`);
                
                ws.onmessage = function(event) {
                    const status = JSON.parse(event.data);
                    
                    if (status.total_chunks > 0) {
                        const progress = (status.chunks_completed / status.total_chunks) * 100;
                        showProgress(`Processing chunk ${status.chunks_completed} of ${status.total_chunks}...`, progress);
                    }
                    
                    if (status.status === 'completed') {
                        hideProgress();
                        showStatus('Streaming synthesis completed!', 'success');
                        ws.close();
                    } else if (status.status === 'error') {
                        hideProgress();
                        showStatus('Synthesis error: ' + (status.error || 'Unknown error'), 'error');
                        ws.close();
                    }
                };
                
                ws.onerror = function(error) {
                    hideProgress();
                    showStatus('WebSocket error: ' + error.message, 'error');
                };
            }
            
            function showProgress(message, percent) {
                document.getElementById('progressSection').style.display = 'block';
                document.getElementById('progressText').textContent = message;
                document.getElementById('progressBar').style.width = percent + '%';
            }
            
            function hideProgress() {
                document.getElementById('progressSection').style.display = 'none';
            }
            
            function showStatus(message, type) {
                const statusSection = document.getElementById('statusSection');
                statusSection.innerHTML = `<div class="status ${type}">${message}</div>`;
                
                // Auto-hide success messages after 5 seconds
                if (type === 'success') {
                    setTimeout(() => {
                        statusSection.innerHTML = '';
                    }, 5000);
                }
            }
            
            // Initialize the interface
            loadVoices();
        </script>
    </body>
    </html>
    """

# Global startup time
startup_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8015)
