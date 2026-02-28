import os
import shutil

# Define your folders
source_folder = '/home/ssppa/GitHub AI/Pores detection/PNGImages'     # Folder with the filenames to match
search_folder = '/home/ssppa/GitHub AI/Pores detection/PedMasks old'     # Folder where matching files might be found
output_folder = '/home/ssppa/GitHub AI/Pores detection/PedMasks'     # Folder to copy matching files into

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get list of filenames to match (without extensions)
source_filenames = set(os.listdir(source_folder))

# Loop through the search folder
for filename in os.listdir(search_folder):
    if filename in source_filenames:
        src_path = os.path.join(search_folder, filename)
        dest_path = os.path.join(output_folder, filename)
        shutil.copy2(src_path, dest_path)
        print(f"✅ Copied: {filename}")

print("🎉 All matching files copied!")