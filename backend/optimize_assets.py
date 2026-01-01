from PIL import Image
import os

def optimize_image(input_path, output_path, quality=80):
    try:
        if not os.path.exists(input_path):
            print(f"Skipping {input_path}, not found.")
            return
            
        with Image.open(input_path) as img:
            # Resize if too huge (limit to 1920px width)
            if img.width > 1920:
                ratio = 1920 / img.width
                new_size = (1920, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
            img.save(output_path, 'WEBP', quality=quality)
            
        original_size = os.path.getsize(input_path) / 1024
        new_size = os.path.getsize(output_path) / 1024
        print(f"Optimized {input_path}: {original_size:.2f}KB -> {new_size:.2f}KB")
        
    except Exception as e:
        print(f"Error optimizing {input_path}: {e}")

assets_dir = 'src/assets'

for root, dirs, files in os.walk(assets_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(root, file)
            filename_no_ext = os.path.splitext(file)[0]
            output_path = os.path.join(root, filename_no_ext + '.webp')
            
            # Skip if webp already exists and is newer
            # if os.path.exists(output_path):
            #      continue
                 
            optimize_image(input_path, output_path)
