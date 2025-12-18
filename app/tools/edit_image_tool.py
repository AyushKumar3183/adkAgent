"""Compute-only image editing tool - returns data, no artifact saving."""
import os
import base64
import mimetypes
from datetime import datetime
from typing import Optional, Dict

import PIL.Image
import google.generativeai as genai
from google.adk.tools import FunctionTool

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the image generation model
image_model = genai.GenerativeModel("gemini-2.5-flash-image-preview")


async def edit_image_compute(
    edit_prompt: str,
    image_path: Optional[str] = None
) -> Dict:
    """
    Compute-only image editing - returns edited image data, does NOT save artifacts.
    
    Args:
        edit_prompt: Text description of desired edits
        image_path: Optional path to image file (relative to uploads or absolute)
    
    Returns:
        Dictionary with:
        - success: bool
        - filename: str
        - mime_type: str
        - message: str
        (NO base64 data to avoid token limits)
    """
    pil_image = None
    mime_type = "image/png"
    original_filename = "unknown.png"
    
    # Load image from file path
    if image_path:
        # Try relative to uploads directory first
        filepath = os.path.join(UPLOAD_DIR, image_path)
        if not os.path.exists(filepath):
            # Try absolute path
            if os.path.exists(image_path):
                filepath = image_path
            else:
                return {
                    "success": False,
                    "error": f"Image file not found: {image_path}",
                    "filename": None,
                    "mime_type": None
                }
        
        try:
            pil_image = PIL.Image.open(filepath)
            original_filename = os.path.basename(filepath)
            mime_type, _ = mimetypes.guess_type(filepath)
            if not mime_type:
                ext = os.path.splitext(filepath)[1].lower()
                mime_map = {
                    '.png': 'image/png',
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.webp': 'image/webp',
                    '.gif': 'image/gif'
                }
                mime_type = mime_map.get(ext, 'image/png')
        except Exception as e:
            return {
                "success": False,
                "error": f"Error loading image: {str(e)}",
                "filename": None,
                "mime_type": None
            }
    else:
        return {
            "success": False,
            "error": "No image_path provided",
            "filename": None,
            "mime_type": None
        }
    
    try:
        content_parts = [pil_image, edit_prompt]
        
        try:
            response = image_model.generate_content(
                content_parts,
                generation_config={"response_modalities": ["TEXT", "IMAGE"]}
            )
        except Exception:
            response = image_model.generate_content(content_parts)
        
        edited_image_bytes = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                for part in parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_data = part.inline_data.data
                        if isinstance(image_data, str):
                            edited_image_bytes = base64.b64decode(image_data)
                        else:
                            edited_image_bytes = image_data
                        break
        
        if not edited_image_bytes:
            return {
                "success": False,
                "error": "Failed to extract edited image from API response.",
                "filename": None,
                "mime_type": None
            }
        
        # Generate filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name, ext = os.path.splitext(original_filename)
        edited_filename = f"edited_{ts}_{name}{ext}"
        
        # Save to disk
        edited_filepath = os.path.join(UPLOAD_DIR, edited_filename)
        with open(edited_filepath, "wb") as f:
            f.write(edited_image_bytes)
        
        # Return response without base64 data to avoid token limits in conversation history
        # Image is saved to disk, so base64 data is not needed in the response
        return {
            "success": True,
            "filename": edited_filename,
            "mime_type": mime_type,
            "message": f"Successfully edited image. Saved as {edited_filename}"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error editing image: {str(e)}",
            "filename": None,
            "mime_type": None
        }


# Create FunctionTool
edit_image_compute_tool = FunctionTool(edit_image_compute)
