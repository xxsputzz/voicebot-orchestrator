"""
AWS Microservices Orchestrator Client
Coordinates between STT, LLM, and TTS microservices running on different AWS machines
"""
import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    host: str
    port: int
    health_path: str = "/health"
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"

class MicroservicesOrchestrator:
    """
    Orchestrates voice processing across multiple AWS microservices
    """
    
    def __init__(self, service_config: Dict[str, ServiceEndpoint]):
        """
        Initialize orchestrator with service endpoints
        
        Args:
            service_config: Dictionary mapping service names to endpoints
        """
        self.services = service_config
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Service health status
        self.service_health = {name: False for name in service_config.keys()}
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "service_response_times": {name: [] for name in service_config.keys()}
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300),  # 5 minute timeout for TTS
            connector=aiohttp.TCPConnector(limit=100)
        )
        await self.health_check_all()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all services"""
        self.logger.info("🏥 Checking health of all microservices...")
        
        tasks = []
        for name, endpoint in self.services.items():
            tasks.append(self._check_service_health(name, endpoint))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(self.services.keys(), results):
            if isinstance(result, Exception):
                self.service_health[name] = False
                self.logger.error(f"❌ {name} health check failed: {result}")
            else:
                self.service_health[name] = result
                status = "✅ Healthy" if result else "❌ Unhealthy"
                self.logger.info(f"{status} - {name}")
        
        healthy_count = sum(self.service_health.values())
        total_count = len(self.service_health)
        self.logger.info(f"🎯 Health check complete: {healthy_count}/{total_count} services healthy")
        
        return self.service_health
    
    async def _check_service_health(self, name: str, endpoint: ServiceEndpoint) -> bool:
        """Check health of a single service"""
        try:
            async with self.session.get(endpoint.health_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy" and data.get("ready", False)
                return False
        except Exception as e:
            self.logger.warning(f"⚠️ {name} health check failed: {e}")
            return False
    
    async def process_voice_pipeline(self, audio_data: bytes, 
                                   conversation_history: Optional[List[Dict]] = None,
                                   tts_engine: str = "auto",
                                   llm_model: str = "mistral") -> Dict[str, Any]:
        """
        Complete voice processing pipeline across microservices
        
        Args:
            audio_data: Raw audio bytes from user
            conversation_history: Previous conversation context
            tts_engine: TTS engine to use ("kokoro", "nari_dia", "auto")
            llm_model: LLM model to use ("mistral", "gpt-oss")
            
        Returns:
            Dictionary with response audio and metadata
        """
        pipeline_start = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Step 1: Speech-to-Text
            self.logger.info("🎙️ Step 1: Converting speech to text...")
            stt_start = time.time()
            
            transcript_result = await self._call_stt_service(audio_data)
            stt_time = time.time() - stt_start
            self.metrics["service_response_times"]["stt"].append(stt_time)
            
            transcript = transcript_result["transcript"]
            self.logger.info(f"📝 Transcript: '{transcript}'")
            
            # Step 2: Language Model Generation
            self.logger.info("🧠 Step 2: Generating response...")
            llm_start = time.time()
            
            llm_result = await self._call_llm_service(
                text=transcript,
                conversation_history=conversation_history,
                model_type=llm_model
            )
            llm_time = time.time() - llm_start
            self.metrics["service_response_times"]["llm"].append(llm_time)
            
            response_text = llm_result["response"]
            self.logger.info(f"💬 Response: '{response_text[:100]}{'...' if len(response_text) > 100 else ''}'")
            
            # Step 3: Text-to-Speech
            self.logger.info(f"🎭 Step 3: Converting text to speech ({tts_engine})...")
            tts_start = time.time()
            
            tts_result = await self._call_tts_service(
                text=response_text,
                engine=tts_engine
            )
            tts_time = time.time() - tts_start
            self.metrics["service_response_times"]["tts"].append(tts_time)
            
            pipeline_time = time.time() - pipeline_start
            
            # Update metrics
            self.metrics["successful_requests"] += 1
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["successful_requests"] - 1) + pipeline_time)
                / self.metrics["successful_requests"]
            )
            
            # Compile results
            result = {
                "success": True,
                "transcript": transcript,
                "response_text": response_text,
                "audio_base64": tts_result["audio_base64"],
                "metadata": {
                    "pipeline_time_seconds": round(pipeline_time, 3),
                    "stt_time_seconds": round(stt_time, 3),
                    "llm_time_seconds": round(llm_time, 3), 
                    "tts_time_seconds": round(tts_time, 3),
                    "stt_metadata": transcript_result,
                    "llm_metadata": llm_result,
                    "tts_metadata": tts_result["metadata"],
                    "services_used": {
                        "stt": self.services["stt"].host,
                        "llm": self.services["llm"].host,
                        "tts": self.services["tts"].host
                    }
                }
            }
            
            self.logger.info(f"✅ Pipeline complete in {pipeline_time:.3f}s")
            return result
            
        except Exception as e:
            self.metrics["failed_requests"] += 1
            pipeline_time = time.time() - pipeline_start
            
            self.logger.error(f"❌ Pipeline failed after {pipeline_time:.3f}s: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "pipeline_time_seconds": round(pipeline_time, 3)
            }
    
    async def _call_stt_service(self, audio_data: bytes) -> Dict[str, Any]:
        """Call STT microservice"""
        if not self.service_health.get("stt", False):
            raise Exception("STT service not healthy")
        
        endpoint = self.services["stt"]
        url = f"{endpoint.base_url}/transcribe_text"
        
        try:
            async with self.session.post(url, data=audio_data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"STT service error {response.status}: {error_text}")
        except Exception as e:
            # Mark service as unhealthy
            self.service_health["stt"] = False
            raise Exception(f"STT service call failed: {e}")
    
    async def _call_llm_service(self, text: str, conversation_history: Optional[List[Dict]] = None,
                              model_type: str = "mistral") -> Dict[str, Any]:
        """Call LLM microservice"""
        if not self.service_health.get("llm", False):
            raise Exception("LLM service not healthy")
        
        endpoint = self.services["llm"]
        url = f"{endpoint.base_url}/generate"
        
        payload = {
            "text": text,
            "model_type": model_type,
            "use_cache": True,
            "domain_context": "banking",
            "conversation_history": conversation_history or []
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"LLM service error {response.status}: {error_text}")
        except Exception as e:
            self.service_health["llm"] = False
            raise Exception(f"LLM service call failed: {e}")
    
    async def _call_tts_service(self, text: str, engine: str = "auto") -> Dict[str, Any]:
        """Call TTS microservice"""
        if not self.service_health.get("tts", False):
            raise Exception("TTS service not healthy")
        
        endpoint = self.services["tts"]
        url = f"{endpoint.base_url}/synthesize"
        
        payload = {
            "text": text,
            "engine": engine,
            "return_audio": True
        }
        
        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"TTS service error {response.status}: {error_text}")
        except Exception as e:
            self.service_health["tts"] = False
            raise Exception(f"TTS service call failed: {e}")
    
    async def get_service_info(self) -> Dict[str, Any]:
        """Get information about all services"""
        info = {}
        
        for name, endpoint in self.services.items():
            try:
                url = f"{endpoint.base_url}/info"
                async with self.session.get(url) as response:
                    if response.status == 200:
                        info[name] = await response.json()
                    else:
                        info[name] = {"error": f"HTTP {response.status}"}
            except Exception as e:
                info[name] = {"error": str(e)}
        
        return info
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        # Calculate average response times per service
        avg_times = {}
        for service, times in self.metrics["service_response_times"].items():
            if times:
                avg_times[service] = sum(times) / len(times)
            else:
                avg_times[service] = 0.0
        
        return {
            "total_requests": self.metrics["total_requests"],
            "successful_requests": self.metrics["successful_requests"],
            "failed_requests": self.metrics["failed_requests"],
            "success_rate": (
                self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1)
            ),
            "average_pipeline_time_seconds": self.metrics["average_response_time"],
            "average_service_times_seconds": avg_times,
            "service_health": self.service_health
        }

# Example AWS configuration
def get_aws_service_config() -> Dict[str, ServiceEndpoint]:
    """
    Configure service endpoints for AWS deployment
    Replace with your actual AWS instance IPs/domains
    """
    return {
        "stt": ServiceEndpoint(
            name="stt",
            host="stt-service.yourdomain.com",  # Or AWS instance IP
            port=8001
        ),
        "llm": ServiceEndpoint(
            name="llm", 
            host="llm-service.yourdomain.com",  # Or AWS instance IP
            port=8002
        ),
        "tts": ServiceEndpoint(
            name="tts",
            host="tts-service.yourdomain.com",  # Or AWS instance IP
            port=8003
        )
    }

# Local development configuration
def get_local_service_config() -> Dict[str, ServiceEndpoint]:
    """
    Configure service endpoints for local development
    All services run on localhost with different ports
    """
    return {
        "stt": ServiceEndpoint(
            name="stt",
            host="localhost",
            port=8001
        ),
        "llm": ServiceEndpoint(
            name="llm", 
            host="localhost",
            port=8002
        ),
        "tts": ServiceEndpoint(
            name="tts",
            host="localhost",
            port=8003
        )
    }

# Example usage
async def example_voice_conversation():
    """Example of using the microservices orchestrator"""
    # Use local config for development, AWS config for production
    config = get_local_service_config()  # Change to get_aws_service_config() for AWS
    
    async with MicroservicesOrchestrator(config) as orchestrator:
        # Check all services are healthy
        health = await orchestrator.health_check_all()
        print(f"Service health: {health}")
        
        # Simulate audio input (replace with real audio bytes)
        fake_audio = b"fake audio data for testing"
        
        # Process voice pipeline
        result = await orchestrator.process_voice_pipeline(
            audio_data=fake_audio,
            tts_engine="kokoro",  # Fast for real-time
            llm_model="mistral"
        )
        
        if result["success"]:
            print(f"✅ Pipeline successful!")
            print(f"Transcript: {result['transcript']}")
            print(f"Response: {result['response_text']}")
            print(f"Total time: {result['metadata']['pipeline_time_seconds']}s")
        else:
            print(f"❌ Pipeline failed: {result['error']}")
        
        # Get performance metrics
        metrics = orchestrator.get_performance_metrics()
        print(f"Performance: {metrics}")

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voicebot Microservices Orchestrator")
    parser.add_argument("--local", action="store_true", help="Use local services (default)")
    parser.add_argument("--aws", action="store_true", help="Use AWS services")
    parser.add_argument("--test", action="store_true", help="Run test conversation")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.test:
        # Run test conversation
        asyncio.run(example_voice_conversation())
    else:
        # Interactive mode
        print("🎭 Voicebot Microservices Orchestrator")
        print("Use --test to run example conversation")
        print("Use --local for localhost services (default)")
        print("Use --aws for AWS services")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    asyncio.run(example_voice_conversation())
