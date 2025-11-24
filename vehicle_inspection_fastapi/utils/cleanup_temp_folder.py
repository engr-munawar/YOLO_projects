import os

def cleanup_temp_folder():
    """Completely empty the temp_images folder"""
    import glob
    try:
        # Get all files in temp_images folder
        files = glob.glob("temp_images/*")
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"🧹 Deleted: {file}")
            except Exception as e:
                print(f"⚠️ Could not delete {file}: {e}")
        
        print(f"✅ Temp folder cleaned up. Deleted {len(files)} files.")
    except Exception as e:
        print(f"❌ Error cleaning temp folder: {e}")