import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API Key from .env
load_dotenv()

app = FastAPI(title="ChatUIBridge API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    text: str
    model_name: str = "llama-3.1-8b-instant"

@app.post("/process")
async def process_text(data: InputData):
    """
    Processes the input text using a dynamically selected Groq LLM.
    """
    try:
        # Create a new LLM instance based on the requested model
        current_llm = ChatGroq(
            model=data.model_name,
            temperature=0.7,
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        response = current_llm.invoke(data.text)
        
        # Return exactly what the model generated
        return {
            "pure_output": response.content if response.content else str(response)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("RELOAD", "true") == "true"
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
    )
