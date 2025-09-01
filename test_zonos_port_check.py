"""
Test Zonos TTS service on different ports to identify the issue
"""
import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Test with different ports
TEST_PORTS = [8014, 8015, 8016, 8017]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[STARTUP] Test service starting on port {app.state.port}")
    yield
    print(f"[SHUTDOWN] Test service stopping on port {app.state.port}")

def create_test_app(port: int):
    app = FastAPI(lifespan=lifespan)
    app.state.port = port
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "port": port, "service": "test-zonos"}
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": f"Test endpoint working on port {port}"}
    
    return app

async def test_port(port: int):
    """Test if a specific port works"""
    print(f"\n🔍 Testing port {port}...")
    try:
        app = create_test_app(port)
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=False
        )
        server = uvicorn.Server(config)
        
        # Start server in background
        import threading
        server_thread = threading.Thread(target=server.run)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait a moment for startup
        await asyncio.sleep(2)
        
        # Test the server
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://localhost:{port}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"✅ Port {port} works! Response: {data}")
                    return True
                else:
                    print(f"❌ Port {port} returned status {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ Port {port} failed: {e}")
        return False

async def main():
    print("🔍 Testing Zonos TTS ports for conflicts...")
    
    working_ports = []
    
    for port in TEST_PORTS:
        success = await test_port(port)
        if success:
            working_ports.append(port)
        await asyncio.sleep(1)  # Wait between tests
    
    print(f"\n📊 Results:")
    print(f"✅ Working ports: {working_ports}")
    if working_ports:
        print(f"🎯 Recommended port for Zonos TTS: {working_ports[0]}")
    else:
        print("❌ No ports working - there's a deeper issue")

if __name__ == "__main__":
    asyncio.run(main())
