"""Compute-only image generation tool - returns data, no artifact saving."""
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

import os
import base64
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import PIL.Image
import google.generativeai as genai
from google.adk.tools import FunctionTool

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable must be set")
genai.configure(api_key=api_key)

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize the image generation model
image_model = genai.GenerativeModel("gemini-2.5-flash-image-preview")


def remove_white_borders(image: PIL.Image.Image, threshold: int = 240) -> PIL.Image.Image:
    """Remove white borders and extra white regions from an image."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    pixels = image.load()
    
    # Find bounding box of non-white content
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            # Check if pixel is not white
            if r < threshold or g < threshold or b < threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    
    # If no non-white content found, return original
    if min_x >= max_x or min_y >= max_y:
        return image
    
    # Crop to bounding box with small padding
    padding = 5
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(width, max_x + padding)
    max_y = min(height, max_y + padding)
    
    return image.crop((min_x, min_y, max_x, max_y))


def split_grid_image(filename: str, remove_original: bool = False) -> Dict:
    """
    Split a 2x2 grid image into 4 separate images.
    
    Args:
        filename: Name of the grid image file
        remove_original: Whether to delete the original grid image after splitting
    
    Returns:
        Dictionary with success status and split image data
    """
    filepath = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(filepath):
        return {
            "success": False,
            "error": f"Image file not found: {filepath}",
            "split_data": []
        }
    
    try:
        # Open the grid image
        grid_image = PIL.Image.open(filepath)
        width, height = grid_image.size
        
        # Calculate quadrant dimensions
        quad_width = width // 2
        quad_height = height // 2
        
        # Extract base name and extension
        base_name, ext = os.path.splitext(filename)
        
        split_data = []  # List of {filename, image_bytes, mime_type}
        
        # Split into 4 quadrants
        for i, (x, y) in enumerate([(0, 0), (quad_width, 0), (0, quad_height), (quad_width, quad_height)], 1):
            # Crop quadrant
            quadrant = grid_image.crop((x, y, x + quad_width, y + quad_height))
            
            # Remove white borders
            quadrant = remove_white_borders(quadrant)
            
            # Generate filename
            quad_filename = f"{base_name}_{i}{ext}"
            quad_filepath = os.path.join(UPLOAD_DIR, quad_filename)
            
            # Save quadrant
            quadrant.save(quad_filepath)
            
            # Read saved image bytes
            with open(quad_filepath, "rb") as f:
                image_bytes = f.read()
            
            split_data.append({
                "filename": quad_filename,
                "image_bytes": base64.b64encode(image_bytes).decode('utf-8'),
                "mime_type": "image/png"
            })
        
        # Remove original if requested
        if remove_original:
            try:
                os.remove(filepath)
            except Exception as e:
                print(f"Warning: Could not remove original file {filepath}: {e}")
        
        return {
            "success": True,
            "split_data": split_data,
            "message": f"Successfully split grid into {len(split_data)} images"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error splitting grid image: {str(e)}",
            "split_data": []
        }


async def generate_image_compute(
    prompt: str,
    pack_size: int = 1
) -> Dict:
    """
    Compute-only image generation - returns image data, does NOT save artifacts.
    
    Args:
        prompt: Text prompt describing the images to generate
        pack_size: Always forced to 1 (generates 4 images)
    
    Returns:
        Dictionary with:
        - success: bool
        - images: List of {filename, mime_type} (NO base64 data to avoid token limits)
        - message: str
    """
    # Force pack_size=1
    if pack_size != 1:
        pack_size = 1
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_id = str(uuid.uuid4())[:8]
    ts = f"{ts}_{unique_id}"
    
    print(f"[generate_image_compute] Starting generation with unique ID: {unique_id}")
    
    all_images = []
    errors = []
    
    for i in range(pack_size):
        try:
            response = image_model.generate_content(prompt)
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    parts = candidate.content.parts
                    for part in parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            image_data = part.inline_data.data
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data
                            
                            # Save temporarily for splitting
                            name = f"image_{ts}_{i+1}.png"
                            filepath = os.path.join(UPLOAD_DIR, name)
                            with open(filepath, "wb") as f:
                                f.write(image_bytes)
                            
                            # Split the grid
                            split_result = split_grid_image(name, remove_original=True)
                            if split_result.get("success"):
                                split_data = split_result.get("split_data", [])
                                all_images.extend(split_data)
                            else:
                                # If splitting failed, return original
                                with open(filepath, "rb") as f:
                                    orig_bytes = f.read()
                                all_images.append({
                                    "filename": name,
                                    "image_bytes": base64.b64encode(orig_bytes).decode('utf-8'),
                                    "mime_type": "image/png"
                                })
                            break
        except Exception as e:
            errors.append(f"Error generating image {i+1}: {str(e)}")
            continue
    
    if not all_images:
        return {
            "success": False,
            "error": "Failed to generate any images. " + ("; ".join(errors) if errors else "Unknown error"),
            "images": []
        }
    
    # Return response without base64 data to avoid token limits in conversation history
    # Images are saved to disk, so base64 data is not needed in the response
    images_lightweight = [
        {
            "filename": img["filename"],
            "mime_type": img.get("mime_type", "image/png")
        }
        for img in all_images
    ]
    
    return {
        "success": True,
        "images": images_lightweight,
        "message": f"Successfully generated {len(all_images)} images",
        "unique_id": unique_id
    }


# Create FunctionTool
generate_image_compute_tool = FunctionTool(generate_image_compute)
