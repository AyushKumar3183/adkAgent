"""Wrapper tools that handle artifact saving for compute-only tools."""
import os
import base64
from typing import Optional, Dict, List
from google.adk.tools import FunctionTool, ToolContext
from google.genai import types
from ..tools.generate_image_tool import generate_image_compute
from ..tools.edit_image_tool import edit_image_compute
from ..session.memory import get_memory

# Uploads directory
UPLOAD_DIR = "uploads"


async def generate_image_with_artifacts(
    prompt: str,
    pack_size: int = 1,
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Generate images and save artifacts - wrapper around compute-only tool.
    
    This tool:
    1. Calls the compute-only generate_image_compute
    2. Saves all returned images as artifacts ONCE
    3. Updates session memory
    4. Returns success message
    """
    # Call compute-only tool
    result = await generate_image_compute(prompt, pack_size)
    
    if not result.get("success"):
        return f"Error: {result.get('error', 'Failed to generate images')}"
    
    images = result.get("images", [])
    if not images:
        return "Error: No images were generated"
    
    # Save artifacts ONCE - read from disk since compute tool no longer returns base64
    saved_count = 0
    memory = get_memory()
    
    if tool_context:
        for image_data in images:
            filename = image_data["filename"]
            mime_type = image_data.get("mime_type", "image/png")
            
            try:
                # Read image from disk (compute tool saves images to disk)
                filepath = os.path.join(UPLOAD_DIR, filename)
                if not os.path.exists(filepath):
                    print(f"Warning: Image file not found: {filepath}")
                    continue
                
                with open(filepath, "rb") as f:
                    image_bytes = f.read()
                
                # Create artifact part
                artifact_part = types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
                
                # Save artifact ONCE
                await tool_context.save_artifact(
                    filename=filename,
                    artifact=artifact_part
                )
                saved_count += 1
                
                # Update memory with first image (most recent)
                if saved_count == 1:
                    memory.set_last_generated(filename)
                
            except Exception as e:
                print(f"Warning: Could not save artifact {filename}: {str(e)}")
    
    return f"Successfully generated {len(images)} images. Artifacts saved: {saved_count}"


async def edit_image_with_artifacts(
    edit_prompt: str,
    image_path: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Edit image and save artifact - wrapper around compute-only tool.
    
    This tool:
    1. Gets image_path from memory if not provided
    2. Calls the compute-only edit_image_compute
    3. Saves returned image as artifact ONCE
    4. Updates session memory
    5. Returns success message
    """
    memory = get_memory()
    
    # If no image_path, try to get from memory
    if not image_path:
        image_path = memory.get_last_generated()
        if image_path:
            print(f"[edit_image_with_artifacts] Using last generated image: {image_path}")
        else:
            # Try most recent image
            image_path = memory.get_most_recent()
            if image_path:
                print(f"[edit_image_with_artifacts] Using most recent image: {image_path}")
    
    if not image_path:
        return "Error: No image_path provided and no recent image found in memory. Please specify an image to edit."
    
    # Call compute-only tool
    result = await edit_image_compute(edit_prompt, image_path)
    
    if not result.get("success"):
        return f"Error: {result.get('error', 'Failed to edit image')}"
    
    filename = result["filename"]
    mime_type = result.get("mime_type", "image/png")
    
    # Save artifact ONCE - read from disk since compute tool no longer returns base64
    if tool_context:
        try:
            # Read image from disk (compute tool saves images to disk)
            filepath = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(filepath):
                return f"Error: Edited image file not found: {filepath}"
            
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            
            # Create artifact part
            artifact_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )
            
            # Save artifact ONCE
            await tool_context.save_artifact(
                filename=filename,
                artifact=artifact_part
            )
            
            # Update memory
            memory.set_last_edited(filename)
            
            return f"Successfully edited image. Saved as {filename}"
            
        except Exception as e:
            return f"Error: Could not save artifact: {str(e)}"
    
    return f"Successfully edited image. Saved as {filename} (artifact saving skipped - no tool_context)"


async def save_image_artifact_from_disk(
    filename: str,
    update_memory: str = "none",
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Load an image from disk and save it as an artifact.
    Used by root agent after sub-agents return filenames.
    
    Args:
        filename: Name of the image file (in uploads directory)
        update_memory: How to update memory - "generated" (set as last generated), 
                      "edited" (set as last edited), or "none" (don't update)
        tool_context: ToolContext for saving artifacts
    
    Returns:
        Success message or error
    """
    if not tool_context:
        return f"Error: No tool_context available to save artifact for {filename}"
    
    # Resolve file path
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        # Try absolute path
        if os.path.exists(filename):
            filepath = filename
        else:
            return f"Error: Image file not found: {filename}"
    
    try:
        # Read image from disk
        with open(filepath, "rb") as f:
            image_bytes = f.read()
        
        # Determine MIME type
        import mimetypes
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
        
        # Create artifact part
        artifact_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )
        
        # Save artifact
        await tool_context.save_artifact(
            filename=filename,
            artifact=artifact_part
        )
        
        # Update memory if requested
        memory = get_memory()
        if update_memory == "generated":
            memory.set_last_generated(filename)
        elif update_memory == "edited":
            memory.set_last_edited(filename)
        
        return f"Successfully saved artifact: {filename}"
        
    except Exception as e:
        return f"Error saving artifact {filename}: {str(e)}"


# Wrapper function with the desired tool name for agent compatibility
async def save_image_artifact_tool(
    filename: str,
    update_memory: str = "none",
    tool_context: Optional[ToolContext] = None
) -> str:
    """
    Load an image from disk and save it as an artifact.
    Used by root agent after sub-agents return filenames.
    
    Args:
        filename: Name of the image file (in uploads directory)
        update_memory: How to update memory - "generated" (set as last generated), 
                      "edited" (set as last edited), or "none" (don't update)
        tool_context: ToolContext for saving artifacts
    
    Returns:
        Success message or error
    """
    return await save_image_artifact_from_disk(filename, update_memory, tool_context)


# Create FunctionTools
generate_image_tool = FunctionTool(generate_image_with_artifacts)
edit_image_tool = FunctionTool(edit_image_with_artifacts)
# Store function reference before creating tool to avoid name collision
_save_tool_func = save_image_artifact_tool
save_image_artifact_tool = FunctionTool(_save_tool_func)
