from dotenv import load_dotenv
load_dotenv() 
from fastapi import FastAPI, Request
import re
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from app.api.routes import ingest, query, documents
import sys

app = FastAPI(title="ContextQuery API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://contextquery-frontend.*\.vercel\.app|http://localhost:3000",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(documents.router)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request data",
            "errors": exc.errors(),
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Unexpected server error: {str(exc)}"},
    )



@app.get("/health")
async def health():
    return {"status": "ok", "service": "contextquery-backend", "python": sys.version.split()[0]}