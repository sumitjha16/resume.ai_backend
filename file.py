from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any  # Added typing imports
import os
import asyncio
import traceback
from datetime import datetime
from main import EnhancedResumeAnalyzer

# Initialize FastAPI app
app = FastAPI(title="Resume Analyzer API", description="Analyze resumes and extract actionable insights.")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resume-analyze-ai.netlify.app",
        "http://resume-analyze-ai.netlify.app",
        "http://localhost:3000",  # If you're testing locally
        "http://localhost:5173"   # For Vite's default port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Initialize analyzer with better error handling
try:
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    analyzer = EnhancedResumeAnalyzer(mistral_api_key)
except Exception as e:
    print(f"Error initializing EnhancedResumeAnalyzer: {str(e)}")
    raise

@app.post("/analyze")
async def analyze_resume(file: UploadFile) -> Dict[str, Any]:
    """Endpoint to analyze a resume PDF with improved error handling and timestamp tracking"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Record start time for processing
        start_time = datetime.now()

        pdf_content = await file.read()

        # Process PDF with timeout
        try:
            raw_text = await asyncio.wait_for(
                asyncio.to_thread(analyzer.extract_text_from_pdf, pdf_content),
                timeout=20.0
            )
            if not raw_text:
                raise ValueError("No text could be extracted from the PDF")
        except asyncio.TimeoutError:
            raise HTTPException(status_code=500, detail="PDF processing timed out")
        except Exception as e:
            print(f"PDF extraction error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"PDF extraction failed: {str(e)}")

        # Process content
        processed_content = analyzer.process_resume_content(raw_text)

        # Perform AI analysis with extended timeout
        try:
            analysis = await asyncio.wait_for(
                analyzer.get_ai_analysis(processed_content),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=500,
                detail="Analysis timed out. Please try again or upload a shorter resume."
            )
        except Exception as e:
            print(f"AI analysis error: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

        # Calculate processing duration
        processing_duration = (datetime.now() - start_time).total_seconds()

        return JSONResponse(content={
            "analysis": analysis,
            "extracted_content": processed_content,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": processing_duration,
                "version": "2.0.0"
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# main.py (add these imports at the top)
