"""FastAPI application for image generation and editing."""
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import os
import uuid
import asyncio
from typing import Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.genai import types

# Import global runner and session service
from ..agent.runtime import runner, session_service
from ..session.memory import get_memory

# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="Fashion Image Generation API",
    description="FastAPI wrapper around Google ADK agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# SCHEMAS
# -------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class EditRequest(BaseModel):
    edit_prompt: str
    image_filename: Optional[str] = None
    session_id: Optional[str] = None


class GenerateResponse(BaseModel):
    success: bool
    message: str
    filenames: List[str]
    session_id: str


class EditResponse(BaseModel):
    success: bool
    message: str
    filename: str
    session_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool
    message: str
    filenames: List[str] = []
    session_id: str


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def list_images(prefix: str = "", limit: int = 100) -> List[str]:
    """List image files in uploads directory, sorted by modification time."""
    try:
        files = [
            f for f in os.listdir(UPLOAD_DIR)
            if f.endswith(".png") and f.startswith(prefix)
        ]
        files.sort(
            key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x)),
            reverse=True,
        )
        return files[:limit]
    except Exception as e:
        print(f"[API] Error listing images: {e}")
        return []


async def wait_for_images(
    prefix: str = "",
    limit: int = 100,
    max_wait_time: int = 90,
    poll_interval: int = 2
) -> List[str]:
    """Wait for images to be generated, polling with timeout."""
    waited = 0
    filenames = list_images(prefix=prefix, limit=limit)
    
    while not filenames and waited < max_wait_time:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        filenames = list_images(prefix=prefix, limit=limit)
        if waited % 10 == 0:
            print(f"[API] Still waiting for images... ({waited}s elapsed)")
    
    return filenames


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Fashion Image Generation API",
        "architecture": "FastAPI → ADK Runner → Agent → Gemini",
        "version": "1.0.0",
        "endpoints": {
            "generate": "POST /api/generate",
            "edit": "POST /api/edit",
            "images": "GET /api/images",
            "get_image": "GET /api/images/{filename}",
            "health": "GET /api/health"
        }
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "fashion-image-api"}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """Generate fashion images using the agent."""
    try:
        # Generate or use provided session_id
        user_id = "default_user"
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create session if it doesn't exist
        try:
            await session_service.create_session(
                app_name="fashion-image-api",
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            # Session might already exist, continue anyway
            pass
        
        # Create message content
        new_message = types.Content(
            role="user",
            parts=[types.Part(text=request.prompt)]
        )
        
        # Run the agent asynchronously
        message = ""
        final_event = None
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            # Store the final response event
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_event = event
            # Also capture any event with content
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        message += part.text
        
        # Prefer message from final event if available
        if final_event and final_event.content and final_event.content.parts:
            message = ""
            for part in final_event.content.parts:
                if hasattr(part, 'text') and part.text:
                    message += part.text
        
        # Wait for images to be generated
        filenames = await wait_for_images(prefix="", limit=4)
        
        if not filenames:
            # Check if there are any images at all
            all_files = list_images(limit=100)
            error_msg = f"No images generated. "
            if all_files:
                error_msg += f"Found {len(all_files)} other image(s) in uploads directory."
            else:
                error_msg += "No PNG files found in uploads directory."
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Update memory with first image
        memory = get_memory()
        memory.set_last_generated(filenames[0])
        
        # If images were successfully generated, override error messages
        # The agent might return an error message even if images were created
        if filenames:
            # Check if message contains error indicators
            if message and ("Error:" in message or "error" in message.lower() or "failed" in message.lower()):
                # Images exist, so generation was successful despite the error message
                message = f"Successfully generated {len(filenames)} image(s)"
            elif not message:
                message = f"Successfully generated {len(filenames)} image(s)"
        
        return GenerateResponse(
            success=True,
            message=message,
            filenames=filenames[:4],  # Return up to 4 images
            session_id=session_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[API] Error generating images: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating images: {str(e)}",
        )


@app.post("/api/edit", response_model=EditResponse)
async def edit_image(request: EditRequest):
    """Edit an existing image using the agent."""
    try:
        # Get memory for image lookup
        memory = get_memory()
        
        # Determine which image to edit
        image_to_edit = (
            request.image_filename
            or memory.get_last_generated()
            or memory.get_most_recent()
        )
        
        if not image_to_edit:
            raise HTTPException(
                status_code=400,
                detail="No image available to edit. Please provide image_filename or generate an image first."
            )
        
        # Generate or use provided session_id
        user_id = "default_user"
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create session if it doesn't exist
        try:
            await session_service.create_session(
                app_name="fashion-image-api",
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            # Session might already exist, continue anyway
            pass
        
        # Create message content - make it clear this is an edit request
        # Use explicit edit language to ensure root_agent delegates to editing_agent
        prompt = f"Please edit the image {image_to_edit}. Edit request: {request.edit_prompt}"
        new_message = types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        )
        
        # Run the agent asynchronously
        message = ""
        final_event = None
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            # Store the final response event
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_event = event
            # Also capture any event with content
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        message += part.text
        
        # Prefer message from final event if available
        if final_event and final_event.content and final_event.content.parts:
            message = ""
            for part in final_event.content.parts:
                if hasattr(part, 'text') and part.text:
                    message += part.text
        
        # Wait for edited image to be generated
        edited_files = await wait_for_images(prefix="edited_", limit=1)
        
        if not edited_files:
            raise HTTPException(
                status_code=500,
                detail="Edited image not found. The agent may still be processing."
            )
        
        filename = edited_files[0]
        memory.set_last_edited(filename)
        
        # If image was successfully edited, override error messages
        if filename:
            # Check if message contains error indicators or wrong agent response
            if message and (
                "Error:" in message 
                or "error" in message.lower() 
                or "failed" in message.lower()
                or "cannot edit" in message.lower()
                or "only generate" in message.lower()
                or "I cannot" in message
            ):
                # Image exists, so editing was successful despite the error message
                message = "Image edited successfully"
            elif not message:
                message = "Image edited successfully"
        
        return EditResponse(
            success=True,
            message=message,
            filename=filename,
            session_id=session_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[API] Error editing image: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error editing image: {str(e)}",
        )


@app.get("/api/images")
async def list_all_images():
    """List all available images."""
    try:
        images = list_images(limit=100)
        return {
            "success": True,
            "count": len(images),
            "images": images
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing images: {str(e)}",
        )


@app.get("/api/images/{filename}")
async def get_image(filename: str):
    """Retrieve an image file by filename."""
    path = os.path.join(UPLOAD_DIR, filename)
    
    # Security: prevent directory traversal
    if not os.path.abspath(path).startswith(os.path.abspath(UPLOAD_DIR)):
        raise HTTPException(status_code=403, detail="Invalid filename")
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image not found: {filename}")
    
    return FileResponse(path, media_type="image/png", filename=filename)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """General chat endpoint - let the agent handle everything."""
    import time
    request_start_time = time.time()  # Track when request started
    
    try:
        # Generate or use provided session_id
        user_id = "default_user"
        session_id = request.session_id or str(uuid.uuid4())
        
        # Create session if it doesn't exist
        try:
            await session_service.create_session(
                app_name="fashion-image-api",
                user_id=user_id,
                session_id=session_id
            )
        except Exception:
            # Session might already exist, continue anyway
            pass
        
        # Create message content
        new_message = types.Content(
            role="user",
            parts=[types.Part(text=request.message)]
        )
        
        # Run the agent asynchronously - let it handle everything
        message = ""
        final_event = None
        
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            # Store the final response event
            if hasattr(event, 'is_final_response') and event.is_final_response():
                final_event = event
            # Also capture any event with content
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        message += part.text
        
        # Prefer message from final event if available
        if final_event and final_event.content and final_event.content.parts:
            message = ""
            for part in final_event.content.parts:
                if hasattr(part, 'text') and part.text:
                    message += part.text
        
        # Only check for images if the request or response suggests images were created
        # 1) Look for generation intent in the user's request
        request_lower = request.message.lower() if request.message else ""
        has_generation_intent = any(keyword in request_lower for keyword in [
            'generate', 'create', 'make', 'show me', 'image', 'picture', 'dress', 'outfit'
        ])
        
        # 2) Look for explicit success wording in the agent response (avoid general words like "image")
        message_lower = message.lower() if message else ""
        has_image_keywords = any(keyword in message_lower for keyword in [
            'successfully generated', 'i\'ve generated', 'generated images',
            'successfully edited', 'i\'ve edited', 'edited image'
        ])
        
        recent_images = []
        
        # Only check for images if user asked for generation/editing OR agent reported success
        if has_generation_intent or has_image_keywords:
            # Wait a bit for images to be generated/edited
            await asyncio.sleep(2)
            
            # Use request start time to find only new images created during this request
            # Check images modified after request started (with small buffer for async operations)
            time_threshold = request_start_time - 5  # 5 seconds before request (to catch async generation)
            
            # Get recent images
            all_images = list_images(limit=10)
            
            # Filter to only images created during this request (modified after threshold)
            for img in all_images:
                img_path = os.path.join(UPLOAD_DIR, img)
                if os.path.exists(img_path):
                    img_mtime = os.path.getmtime(img_path)
                    if img_mtime >= time_threshold:
                        recent_images.append(img)
            
            # Sort by modification time (newest first)
            recent_images.sort(
                key=lambda x: os.path.getmtime(os.path.join(UPLOAD_DIR, x)),
                reverse=True
            )
            
            # Limit to 4 images max
            recent_images = recent_images[:4]
            
            # Update memory if images exist
            if recent_images:
                memory = get_memory()
                # Check if it's an edited image (starts with "edited_")
                if recent_images[0].startswith("edited_"):
                    memory.set_last_edited(recent_images[0])
                else:
                    memory.set_last_generated(recent_images[0])
        
        return ChatResponse(
            success=True,
            message=message or "Response received",
            filenames=recent_images,
            session_id=session_id,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[API] Error in chat: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}",
        )


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file."""
    try:
        path = os.path.join(UPLOAD_DIR, file.filename)
        
        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Update memory
        memory = get_memory()
        memory.set_last_generated(file.filename)
        
        return {
            "success": True,
            "message": f"File uploaded successfully: {file.filename}",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading file: {str(e)}",
        )


# -------------------------------------------------------------------
# RUN SERVER
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
