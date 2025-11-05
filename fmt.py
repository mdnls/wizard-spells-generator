import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import io
from typing import Dict, Any, Tuple
import json
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

client = genai.Client(api_key=os.getenv('GEMINI_KEY'))

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def find_color_rectangles(mask_img: Image.Image, target_color: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Find the bounding rectangle of a specific color in the mask image."""
    mask_array = np.array(mask_img)
    
    # Handle both RGBA and RGB images
    if mask_array.shape[2] == 4:  # RGBA
        # Find pixels that match the target color and are not transparent
        matches = np.all(mask_array[:, :, :3] == target_color, axis=2) & (mask_array[:, :, 3] > 0)
    else:  # RGB
        matches = np.all(mask_array == target_color, axis=2)
    
    if not np.any(matches):
        return None
    
    # Find bounding box
    coords = np.where(matches)
    top, left = coords[0].min(), coords[1].min()
    bottom, right = coords[0].max(), coords[1].max()
    
    return (left, top, right + 1, bottom + 1)  # PIL uses (left, top, right, bottom)


def wrap_text(text: str, font, max_width: int, draw) -> list:
    """Wrap text to fit within max_width, returning a list of lines."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        # Test if adding this word would exceed the width
        test_line = ' '.join(current_line + [word])
        try:
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
        except:
            test_width = len(test_line) * 10  # Fallback estimation
        
        if test_width <= max_width:
            current_line.append(word)
        else:
            # If current_line is empty, the single word is too long
            if not current_line:
                current_line = [word]  # Force it on its own line
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def calculate_multiline_dimensions(lines: list, font, draw, line_spacing: float = 1.2) -> tuple:
    """Calculate total dimensions for multiple lines of text."""
    if not lines:
        return 0, 0
    
    max_width = 0
    total_height = 0
    
    for i, line in enumerate(lines):
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            line_height = bbox[3] - bbox[1]
        except:
            line_width = len(line) * 8  # Fallback estimation
            line_height = 12
        
        max_width = max(max_width, line_width)
        
        if i == 0:
            total_height = line_height
        else:
            total_height += int(line_height * line_spacing)
    
    return max_width, total_height


def render_text_image(text: str, width: int, height: int, max_font_size: int = None) -> Image.Image:
    """Render text as an image that fits within the given dimensions with maximum font size and multiple lines."""
    # Create image with transparent background
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Set maximum font size if not provided
    if max_font_size is None:
        max_font_size = max(width, height)
    
    # Try to load a font, fall back to default if not available
    def get_font(size):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Luminari.ttf", size)
        except (OSError, IOError):
            try:
                return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
            except (OSError, IOError):
                try:
                    return ImageFont.load_default()
                except:
                    return None
    
    # Binary search for the maximum font size that fits with line wrapping
    min_size = 1
    max_size = max_font_size
    best_font_size = min_size
    best_lines = [text]  # Fallback to single line
    
    while min_size <= max_size:
        current_size = (min_size + max_size) // 2
        font = get_font(current_size)
        
        if font is None:
            break
        
        try:
            # Try wrapping text at this font size
            lines = wrap_text(text, font, width, draw)
            text_width, text_height = calculate_multiline_dimensions(lines, font, draw)
            
            # Check if text fits with line wrapping
            if text_width <= width and text_height <= height:
                best_font_size = current_size
                best_lines = lines
                min_size = current_size + 1
            else:
                max_size = current_size - 1
        except:
            max_size = current_size - 1
    
    # Use the best font size found
    font = get_font(best_font_size)
    if font is None:
        draw.text((5, height//2), text, fill=(0, 0, 0))
        return img
    
    # Draw the text lines
    try:
        # Calculate total text block dimensions
        total_width, total_height = calculate_multiline_dimensions(best_lines, font, draw)
        
        # Start position (centered)
        start_x = (width - total_width) // 2
        start_y = (height - total_height) // 2 - 6  # Raise baseline by 6 pixels
        
        current_y = start_y
        line_spacing = 1.2
        
        for i, line in enumerate(best_lines):
            # Get line dimensions for centering each line individually
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
            except:
                line_width = len(line) * 8
                line_height = 12
            
            # Center each line horizontally
            line_x = (width - line_width) // 2
            
            draw.text((line_x, current_y), line, fill=(0, 0, 0), font=font)
            
            # Move to next line
            if i < len(best_lines) - 1:  # Don't add spacing after last line
                current_y += int(line_height * line_spacing)
                
    except Exception as e:
        # Fallback positioning if anything fails
        draw.text((5, height//2 - 6), text, fill=(0, 0, 0), font=font)
    
    return img


def remove_white_border(img: Image.Image, tolerance: int = 10) -> Image.Image:
    """Remove white borders from an image by detecting and cropping them out."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array for easier processing
    img_array = np.array(img)
    
    # Define what we consider "white" (with tolerance for slight variations)
    def is_white_pixel(pixel):
        return all(channel >= 255 - tolerance for channel in pixel)
    
    # Find the bounds of non-white content
    height, width = img_array.shape[:2]
    
    # Find top border
    top = 0
    for y in range(height):
        if not all(is_white_pixel(img_array[y, x]) for x in range(width)):
            top = y
            break
    
    # Find bottom border
    bottom = height - 1
    for y in range(height - 1, -1, -1):
        if not all(is_white_pixel(img_array[y, x]) for x in range(width)):
            bottom = y
            break
    
    # Find left border
    left = 0
    for x in range(width):
        if not all(is_white_pixel(img_array[y, x]) for y in range(height)):
            left = x
            break
    
    # Find right border
    right = width - 1
    for x in range(width - 1, -1, -1):
        if not all(is_white_pixel(img_array[y, x]) for y in range(height)):
            right = x
            break
    
    # If we found valid bounds, crop the image
    if top <= bottom and left <= right:
        cropped = img.crop((left, top, right + 1, bottom + 1))
        print(f"Removed white border: original {width}x{height} -> cropped {right+1-left}x{bottom+1-top}")
        return cropped
    
    # If no border detected or invalid bounds, return original
    return img


def generate_ai_image(prompt: str, width: int, height: int) -> Image.Image:
    """Generate an AI image using the Gemini API (placeholder implementation)."""
    # This is a placeholder implementation
    # In a real implementation, you would make an API call to Gemini/Nano Banana
    # For now, we'll create a colored rectangle with the prompt text
    
    print(f"Generating AI image for prompt: '{prompt}'")
    
    prompt = f"Make an illustration for a 'spell card' for a wizard themed game. Generate a image to appear on the card, representing the spell. The card has a magic the gathering type layout. Your image will appear in a window on the upper half of the card and does not need to include any text, card description, or bordering, it is just an illustration. Do not return any text or anything but the image itself. Return a square image and use the entire space with your illustration. The prompt is: {prompt} --width {width} --height {height}"

    try: 
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt],
        )

        for part in response.candidates[0].content.parts:
            card_image = Image.open(BytesIO(part.inline_data.data))
            
            # Remove white borders from the generated image
            card_image = remove_white_border(card_image)
            
            return card_image
    except:
        print(f"Could not generate card.") 
    


def fill_mask(img, mask_img, contents_dict):
    '''
    You will receive two images and a dictionary of the following form. 

    example_contents_dict = {
        "ff0000": {"type": "text", "contents": "", "z_index": 1} ,
        "00ff00": {"type": "ai_img", "prompt": "", "z_index": -1} ,
    }

    The first image is what you will modify. The second image, `mask_img,` contains only transparent pixels
    and rectangular shapes of consistent color. Using the input dictionary and the mask, you will modify
    the target image in the following way. 
    1. For an entry in `contents_dict`, use the key to identify the target color within the mask. 
    2. Compute the size and location of the masking rectangle in mask image. 
    3. If the `contents_dict` is of type text, render an image of the text contained in the "contents" key for this entry.
       If the `contents_dict` is of type "ai_image", use the Gemini API to make a request to the Nano Banana model with the given prompt. 
           Download the response image. Rescale and crop to fit in the masking rectangle. 
    4. Scale this image to fit within masking rectangle. 
    5. Repeat these steps for every entry in contents_dict
    6. Finally, compose these images according to the given z_index. Items with the lower z_index are painted
    onto the canvas first, and target_image is assumed to have z_index zero. If two items have the same z_index,
    paint them in the order they appear. 
    '''
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    if not isinstance(mask_img, Image.Image):
        mask_img = Image.open(mask_img)
    
    # Convert to RGBA for proper compositing
    result_img = img.convert('RGBA')
    
    # Collect all items to be composited with their z-indices
    composite_items = []
    
    # Process each entry in contents_dict
    for order, (color_key, content_info) in enumerate(contents_dict.items()):
        # Convert hex color to RGB
        target_color = hex_to_rgb(color_key)
        
        # Find the rectangle for this color in the mask
        rect = find_color_rectangles(mask_img, target_color)
        if rect is None:
            print(f"Warning: Color {color_key} not found in mask image")
            continue
        
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        
        # Generate content based on type
        if content_info["type"] == "text":
            max_font_size = content_info.get("max_font_size", None)
            content_img = render_text_image(
                content_info["contents"], 
                width, 
                height,
                max_font_size
            )
        elif content_info["type"] == "ai_img":
            content_img = generate_ai_image(
                content_info["prompt"], 
                width, 
                height
            )
            if content_img is None:
                return
        else:
            print(f"Warning: Unknown content type {content_info['type']}")
            continue
        
        # Ensure content image fits the rectangle
        if content_img.size != (width, height):
            content_img = content_img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Ensure content image is in RGBA format for proper transparency handling
        if content_img.mode != 'RGBA':
            content_img = content_img.convert('RGBA')
        
        # Add to composite list with position, z-index, and order
        composite_items.append({
            'image': content_img,
            'position': (left, top),
            'z_index': content_info.get('z_index', 0),
            'order': order
        })
    
    # Sort by z-index (lower z-index drawn first, then by order of appearance for same z-index)
    composite_items.sort(key=lambda x: (x['z_index'], x.get('order', 0)))
    
    # Debug: Print z-index order
    print("Compositing order (z-index):")
    for i, item in enumerate(composite_items):
        print(f"  {i}: z_index={item['z_index']}")
    
    # Composite all items onto the result image
    for item in composite_items:
        # Create a temporary image for compositing
        temp_img = Image.new('RGBA', result_img.size, (0, 0, 0, 0))
        temp_img.paste(item['image'], item['position'])
        
        # Composite with the result
        result_img = Image.alpha_composite(result_img, temp_img)
    
    return result_img


if __name__ == "__main__":
    # Set up paths
    rare_card_template_img = "template images/rare.png"
    common_card_template_img = "template images/common.png"
    salamancer_card_template_img = "template images/salamancer.png"
    pyromancer_card_template_img = "template images/pyromancer.png"
    alchemancer_card_template_img = "template images/alchemancer.png"
    aquamancer_card_template_img = "template images/aquamancer.png"
    romancer_card_template_img = "template images/romancer.png"
    rare_mask_img = "template images/rare-mask.png"
    common_mask_img = "template images/common-mask.png"
    spells_tsv = "reward spells.tsv"

    # Example usage
    example_contents_dict = {
        "ff0000": {"type": "ai_img", "prompt": "magical lightning bolt spell effect", "z_index": -1},
        "00ff00": {"type": "text", "contents": "Lightning Bolt", "z_index": 1, "max_font_size": 30},
        "0000ff": {"type": "text", "contents": "Shoot a bolt of lightning", "z_index": 1, "max_font_size": 30}
    }

    card_manifest = [

    ]

    # Create cards directory if it doesn't exist
    os.makedirs('cards', exist_ok=True)
    
    with open('reward spells extended.tsv', 'r') as f_in:
        data = [line.split("\t") for line in f_in.readlines()]
    
    for spell in data[1:]:
        rarity = spell[4]
        name = spell[5]
        desc = spell[6] 
        prompt_help = spell[7]
        
        # Check if card already exists
        card_path = f'cards/{name}.png'
        if os.path.exists(card_path):
            print(f"Skipping {name} - already exists")
            continue
        
        prompt = f"{name}: {desc}. {prompt_help}"
        if name == "Little Piss Boy":
            prompt = f"Weewee Boy: a young wizard is standing in a long bathroom line who really has to pee. The other wizards are laughing at him."
        card_manifest.append(
            {
                "ff0000": {"type": "ai_img", "prompt": prompt, "z_index": -1},
                "00ff00": {"type": "text", "contents": name, "z_index": 1, "max_font_size": 30},
                "0000ff": {"type": "text", "contents": desc, "z_index": 1, "max_font_size": 30}
            })
        
        print(f"Generating card: {name}")
        
        if rarity == "rare":
            card_img = fill_mask(rare_card_template_img, rare_mask_img, card_manifest[-1]) 
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "common":
            card_img = fill_mask(common_card_template_img, common_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "Salamancer":
            card_img = fill_mask(salamancer_card_template_img, rare_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "Aquamancer":
            card_img = fill_mask(aquamancer_card_template_img, rare_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "Romancer":
            card_img = fill_mask(romancer_card_template_img, rare_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "Pyromancer":
            card_img = fill_mask(pyromancer_card_template_img, rare_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        elif rarity == "Alchemancer":
            card_img = fill_mask(alchemancer_card_template_img, rare_mask_img, card_manifest[-1])
            if card_img is None:
                continue
            card_img.save(card_path)
        else: 
            print(f"Invalid card – {prompt}")

    
