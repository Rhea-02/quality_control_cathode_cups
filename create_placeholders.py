# Placeholder Image Generator
# This creates simple colored placeholder images for the dashboard
from PIL import Image, ImageDraw, ImageFont
import os

def create_placeholder(filename, color, text, size=(80, 80)):
    """Create a colored placeholder image with text"""
    img = Image.new('RGB', size, color=color)
    draw = ImageDraw.Draw(img)
    
    # Add text in center
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        # Fall back to default font
        font = ImageFont.load_default()
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    
    draw.text(position, text, fill='white', font=font)
    img.save(filename)
    print(f"Created: {filename}")

# Create assets directory placeholders
assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
os.makedirs(assets_dir, exist_ok=True)

# Create placeholder cathode cup images
create_placeholder(
    os.path.join(assets_dir, 'cathode_cup.png'),
    '#4B5563',
    'Cathode\nCup',
    (80, 80)
)

print("\nPlaceholder images created successfully!")
print("You can replace these with real images from your output folder.")
