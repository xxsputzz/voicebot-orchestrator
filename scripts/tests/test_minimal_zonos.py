#!/usr/bin/env python3
"""Minimal Zonos TTS service test to isolate issues."""

from fastapi import FastAPI
import uvicorn
import logging

app = FastAPI(title="Test Zonos TTS")

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "test_zonos"}

@app.get("/voices") 
async def get_voices():
    """Return simple voice list"""
    return ["default", "professional", "conversational", "narrative"]

@app.get("/models")
async def get_models():
    """Return simple model list"""  
    return ["zonos-v1", "zonos-v2", "zonos-lite"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("[TEST] Starting minimal Zonos TTS test service...")
    uvicorn.run(app, host="0.0.0.0", port=8015, log_level="info")
