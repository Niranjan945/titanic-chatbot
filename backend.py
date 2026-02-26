from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import TitanicChatAgent
import uvicorn
from contextlib import asynccontextmanager

# Initialize the agent (singleton pattern)
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on server startup"""
    global agent
    print("🚀 Starting Titanic Chat API...")
    agent = TitanicChatAgent()
    print("✅ API ready to receive requests!")
    yield
    # Clean up code can go here if needed when shutting down

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Chat API",
    description="API for querying Titanic dataset with natural language",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allows Streamlit to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    visualization: str | None = None
    error: str | None = None

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Titanic Chat API is running",
        "endpoints": {
            "POST /query": "Submit a question about Titanic dataset",
            "GET /info": "Get dataset information",
            "GET /docs": "Interactive API documentation"
        }
    }

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_titanic(request: QueryRequest):
    """Process natural language query about Titanic dataset"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        print(f"\n📥 Received query: {request.question}")
        result = agent.query(request.question)
        
        if result['error']:
            print(f"❌ Error: {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])
        
        print(f"✅ Response generated successfully")
        return QueryResponse(
            answer=result['answer'],
            visualization=result['visualization'],
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Dataset info endpoint
@app.get("/info")
async def get_dataset_info():
    """Get basic information about the Titanic dataset"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        info = agent.get_dataset_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset info: {str(e)}")

# Run the server (only when executed directly)
if __name__ == "__main__":
    print("🚢 Starting Titanic Chat Backend Server...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )