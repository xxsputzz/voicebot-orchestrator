"""
Independent Microservices Orchestrator Client
Coordinates calls to separate TTS and LLM services
"""
import aiohttp
import asyncio
import logging
import time
import base64
from typing import Dict, Any, Optional, List
import json

class IndependentServicesOrchestrator:
    """
    Orchestrator for independent TTS and LLM microservices
    """
    
    def __init__(self, service_config: Dict[str, str]):
        """
        Initialize with service endpoints
        
        Args:
            service_config: Dictionary with service URLs
                {
                    "stt": "http://localhost:8001",
                    "kokoro_tts": "http://localhost:8011", 
                    "hira_dia_tts": "http://localhost:8012",
                    "mistral_llm": "http://localhost:8021",
                    "gpt_llm": "http://localhost:8022"
                }
        """
        self.config = service_config
        self.session = None
        self.performance_metrics = {
            "requests_made": 0,
            "total_time": 0,
            "average_time": 0,
            "service_calls": {}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Check health of all configured services"""
        health_results = {}
        
        for service_name, url in self.config.items():
            try:
                async with self.session.get(f"{url}/health", timeout=10) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        health_results[service_name] = {
                            "status": "healthy",
                            "url": url,
                            "details": health_data
                        }
                    else:
                        health_results[service_name] = {
                            "status": "unhealthy",
                            "url": url,
                            "status_code": response.status
                        }
            except Exception as e:
                health_results[service_name] = {
                    "status": "error",
                    "url": url,
                    "error": str(e)
                }
        
        return health_results
    
    async def transcribe_audio(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio using STT service"""
        if "stt" not in self.config:
            raise ValueError("STT service not configured")
        
        start_time = time.time()
        
        try:
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('audio', audio_data, filename='audio.wav', content_type='audio/wav')
            
            async with self.session.post(
                f"{self.config['stt']}/transcribe", 
                data=data,
                timeout=30
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    self._update_metrics("stt", processing_time)
                    
                    return {
                        "success": True,
                        "transcript": result.get("transcript", ""),
                        "processing_time": processing_time,
                        "metadata": result.get("metadata", {})
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"STT failed with status {response.status}: {error_text}"
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"STT service error: {str(e)}"
            }
    
    async def generate_llm_response(
        self, 
        text: str, 
        model: str = "mistral",
        conversation_history: Optional[List[Dict]] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate response using specified LLM service"""
        
        # Determine which LLM service to use
        if model.lower() in ["mistral"]:
            service_key = "mistral_llm"
        elif model.lower() in ["gpt", "gpt-oss"]:
            service_key = "gpt_llm"
        else:
            return {
                "success": False,
                "error": f"Unknown LLM model: {model}. Available: mistral, gpt"
            }
        
        if service_key not in self.config:
            return {
                "success": False,
                "error": f"{service_key} service not configured"
            }
        
        start_time = time.time()
        
        try:
            payload = {
                "text": text,
                "use_cache": use_cache,
                "conversation_history": conversation_history
            }
            
            async with self.session.post(
                f"{self.config[service_key]}/generate",
                json=payload,
                timeout=60
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    self._update_metrics(service_key, processing_time)
                    
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "model_used": result.get("model_used", model),
                        "cache_hit": result.get("cache_hit", False),
                        "processing_time": processing_time,
                        "tokens_generated": result.get("tokens_generated", 0)
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"LLM {model} failed with status {response.status}: {error_text}"
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM {model} service error: {str(e)}"
            }
    
    async def synthesize_speech(
        self, 
        text: str, 
        engine: str = "kokoro",
        voice: str = "af_bella"
    ) -> Dict[str, Any]:
        """Synthesize speech using specified TTS engine"""
        
        # Determine which TTS service to use
        if engine.lower() in ["kokoro"]:
            service_key = "kokoro_tts"
        elif engine.lower() in ["hira_dia", "nari_dia", "hira", "dia"]:
            service_key = "hira_dia_tts"
        else:
            return {
                "success": False,
                "error": f"Unknown TTS engine: {engine}. Available: kokoro, hira_dia"
            }
        
        if service_key not in self.config:
            return {
                "success": False,
                "error": f"{service_key} service not configured"
            }
        
        start_time = time.time()
        
        try:
            payload = {
                "text": text,
                "voice": voice,
                "return_audio": True
            }
            
            async with self.session.post(
                f"{self.config[service_key]}/synthesize",
                json=payload,
                timeout=300  # Longer timeout for Hira Dia
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    processing_time = time.time() - start_time
                    
                    self._update_metrics(service_key, processing_time)
                    
                    # Decode audio if present
                    audio_bytes = None
                    if result.get("audio_base64"):
                        audio_bytes = base64.b64decode(result["audio_base64"])
                    
                    return {
                        "success": True,
                        "audio_data": audio_bytes,
                        "engine_used": engine,
                        "processing_time": processing_time,
                        "metadata": result.get("metadata", {})
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"TTS {engine} failed with status {response.status}: {error_text}"
                    }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"TTS {engine} service error: {str(e)}"
            }
    
    async def process_voice_pipeline(
        self,
        audio_data: bytes,
        tts_engine: str = "kokoro",
        llm_model: str = "mistral",
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process complete voice pipeline with specified engines
        
        Args:
            audio_data: Input audio bytes
            tts_engine: "kokoro" or "hira_dia"
            llm_model: "mistral" or "gpt"
            conversation_history: Previous conversation turns
        """
        pipeline_start = time.time()
        
        try:
            # Step 1: Speech-to-Text
            logging.info("🎤 Starting STT...")
            stt_result = await self.transcribe_audio(audio_data)
            
            if not stt_result["success"]:
                return {
                    "success": False,
                    "error": f"STT failed: {stt_result['error']}",
                    "step": "stt"
                }
            
            transcript = stt_result["transcript"]
            logging.info(f"📝 Transcript: {transcript}")
            
            # Step 2: Language Model
            logging.info(f"🧠 Generating response with {llm_model}...")
            llm_result = await self.generate_llm_response(
                text=transcript,
                model=llm_model,
                conversation_history=conversation_history
            )
            
            if not llm_result["success"]:
                return {
                    "success": False,
                    "error": f"LLM failed: {llm_result['error']}",
                    "step": "llm",
                    "transcript": transcript
                }
            
            response_text = llm_result["response"]
            logging.info(f"💭 Response: {response_text}")
            
            # Step 3: Text-to-Speech
            logging.info(f"🔊 Synthesizing with {tts_engine}...")
            tts_result = await self.synthesize_speech(
                text=response_text,
                engine=tts_engine
            )
            
            if not tts_result["success"]:
                return {
                    "success": False,
                    "error": f"TTS failed: {tts_result['error']}",
                    "step": "tts",
                    "transcript": transcript,
                    "response_text": response_text
                }
            
            total_time = time.time() - pipeline_start
            
            return {
                "success": True,
                "transcript": transcript,
                "response_text": response_text,
                "audio_data": tts_result["audio_data"],
                "metadata": {
                    "pipeline_time_seconds": round(total_time, 3),
                    "stt_time": round(stt_result["processing_time"], 3),
                    "llm_time": round(llm_result["processing_time"], 3),
                    "llm_model": llm_result["model_used"],
                    "llm_cache_hit": llm_result["cache_hit"],
                    "tts_time": round(tts_result["processing_time"], 3),
                    "tts_engine": tts_result["engine_used"],
                    "services_used": {
                        "stt": self.config.get("stt"),
                        "llm": llm_model,
                        "tts": tts_engine
                    }
                }
            }
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            logging.error(f"❌ Pipeline failed after {total_time:.3f}s: {e}")
            return {
                "success": False,
                "error": f"Pipeline error: {str(e)}",
                "step": "pipeline"
            }
    
    def _update_metrics(self, service: str, processing_time: float):
        """Update performance metrics"""
        self.performance_metrics["requests_made"] += 1
        self.performance_metrics["total_time"] += processing_time
        self.performance_metrics["average_time"] = (
            self.performance_metrics["total_time"] / self.performance_metrics["requests_made"]
        )
        
        if service not in self.performance_metrics["service_calls"]:
            self.performance_metrics["service_calls"][service] = {
                "count": 0,
                "total_time": 0,
                "average_time": 0
            }
        
        service_metrics = self.performance_metrics["service_calls"][service]
        service_metrics["count"] += 1
        service_metrics["total_time"] += processing_time
        service_metrics["average_time"] = service_metrics["total_time"] / service_metrics["count"]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    async def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Get information about a specific service"""
        if service_name not in self.config:
            return {"error": f"Service {service_name} not configured"}
        
        try:
            async with self.session.get(f"{self.config[service_name]}/info", timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Failed to get info: {response.status}"}
        except Exception as e:
            return {"error": f"Service info error: {str(e)}"}
    
    async def get_all_service_info(self) -> Dict[str, Any]:
        """Get information about all configured services"""
        info = {}
        for service_name in self.config.keys():
            info[service_name] = await self.get_service_info(service_name)
        return info

def get_local_service_config() -> Dict[str, str]:
    """Get local service configuration"""
    return {
        "stt": "http://localhost:8001",
        "kokoro_tts": "http://localhost:8011",
        "hira_dia_tts": "http://localhost:8012", 
        "mistral_llm": "http://localhost:8021",
        "gpt_llm": "http://localhost:8022"
    }

def get_aws_service_config() -> Dict[str, str]:
    """Get AWS service configuration"""
    return {
        "stt": "https://stt-service.your-domain.com",
        "kokoro_tts": "https://kokoro-tts.your-domain.com",
        "hira_dia_tts": "https://hira-dia-tts.your-domain.com",
        "mistral_llm": "https://mistral-llm.your-domain.com", 
        "gpt_llm": "https://gpt-llm.your-domain.com"
    }

# Example usage
async def example_independent_services():
    """Example of using independent services"""
    config = get_local_service_config()
    
    async with IndependentServicesOrchestrator(config) as orchestrator:
        # Check all services are healthy
        health = await orchestrator.health_check_all()
        print("🏥 Service Health:")
        for service, status in health.items():
            print(f"  {service}: {status['status']}")
        
        # Get service information
        info = await orchestrator.get_all_service_info()
        print("\n📋 Service Info:")
        for service, details in info.items():
            if "error" not in details:
                print(f"  {service}: {details.get('description', 'N/A')}")
        
        # Test different engine combinations
        test_combinations = [
            {"tts": "kokoro", "llm": "mistral"},
            {"tts": "hira_dia", "llm": "gpt"},
            {"tts": "kokoro", "llm": "gpt"},
            {"tts": "hira_dia", "llm": "mistral"}
        ]
        
        fake_audio = b"fake audio data for testing"
        
        for combo in test_combinations:
            print(f"\n🧪 Testing {combo['tts']} TTS + {combo['llm']} LLM...")
            
            result = await orchestrator.process_voice_pipeline(
                audio_data=fake_audio,
                tts_engine=combo["tts"],
                llm_model=combo["llm"]
            )
            
            if result["success"]:
                print(f"✅ Success! Total time: {result['metadata']['pipeline_time_seconds']}s")
                print(f"   STT: {result['metadata']['stt_time']}s")
                print(f"   LLM ({combo['llm']}): {result['metadata']['llm_time']}s") 
                print(f"   TTS ({combo['tts']}): {result['metadata']['tts_time']}s")
            else:
                print(f"❌ Failed: {result['error']}")
        
        # Show performance metrics
        metrics = orchestrator.get_performance_metrics()
        print(f"\n📊 Performance Summary:")
        print(f"  Total requests: {metrics['requests_made']}")
        print(f"  Average time: {metrics['average_time']:.3f}s")
        
        for service, stats in metrics['service_calls'].items():
            print(f"  {service}: {stats['count']} calls, avg {stats['average_time']:.3f}s")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Independent Microservices Orchestrator")
    parser.add_argument("--local", action="store_true", help="Use local services (default)")
    parser.add_argument("--aws", action="store_true", help="Use AWS services") 
    parser.add_argument("--test", action="store_true", help="Run test with all combinations")
    parser.add_argument("--tts", choices=["kokoro", "hira_dia"], default="kokoro", help="TTS engine")
    parser.add_argument("--llm", choices=["mistral", "gpt"], default="mistral", help="LLM model")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        # Run comprehensive test
        asyncio.run(example_independent_services())
    else:
        print(f"🎭 Independent Microservices Orchestrator")
        print(f"TTS Engine: {args.tts}")
        print(f"LLM Model: {args.llm}")
        print("Use --test to run comprehensive testing")

if __name__ == "__main__":
    main()
