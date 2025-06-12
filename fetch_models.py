import urllib.request
import os
import sys

# stuff to grab (this script was from when I didn't have a way to download the models in-app so I made this script to download them to transfer to my vita through ftp)

MODELS = {
    "260K": [
        {
            "name": "stories260K.bin",
            "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin"
        },
        {
            "name": "tok512.bin",
            "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin"
        }
    ],
    "15M": [
        {
            "name": "stories15M.bin",
            "url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin"
        },
        {
            "name": "tokenizer.bin",
            "url": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin"
        }
    ]
}

def _reporthook(count, block_size, total_size):
    """A simple reporthook for urllib.request.urlretrieve."""
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\r    Downloading... {percent}%")
    sys.stdout.flush()

def download_file(name, url):
    """Downloads a file from a URL to the current directory."""
    if os.path.exists(name):
        print(f"-> {name} already exists. Skipping.")
        return
    print(f"-> Downloading {name}...")
    try:
        urllib.request.urlretrieve(url, name, _reporthook)
        sys.stdout.write("\n") # Move to the next line after download completes
        print(f"   Done.")
    except Exception as e:
        print(f"\n   Error downloading {name}: {e}")
        print("   Please check your internet connection and the URL.")

def download_model_suite(key):
    """Downloads all files for a given model suite."""
    if key in MODELS:
        print(f"\nFetching {key} model suite...")
        for file_info in MODELS[key]:
            download_file(file_info["name"], file_info["url"])
    else:
        print(f"Error: Model suite '{key}' not found.")

def display_final_instructions():
    """Prints the final instructions for the user."""
    print("\n" + "="*50)
    print("      DOWNLOAD COMPLETE!")
    print("="*50)
    print("\nThe necessary files have been downloaded to the current directory:")
    print(f"   {os.getcwd()}")
    print("\nNext steps:")
    print("1. Connect your PS Vita to another device via USB or FTP using VitaShell.")
    print("2. Navigate to the `ux0:` partition on your Vita.")
    print("3. Copy the downloaded .bin files into the `ux0:data/` folder.")
    print("\nOnce the files are in place, you can run PSVita-LLM!")
    print("="*50 + "\n")


def main():
    """Main function to run the interactive downloader."""
    print("="*50)
    print("Model Downloader")
    print("="*50)
    print("This script will download the required model and tokenizer")
    print("files for the PSVita-LLM application.\n")

    while True:
        print("Please choose an option:")
        print("  [1] Download the 260K model (~1 MB)")
        print("  [2] Download the 15M model (~60 MB)")
        print("  [3] Download BOTH models")
        print("  [4] Exit")
        
        choice = input("\nEnter your choice (1-4): ")

        if choice == '1':
            download_model_suite("260K")
            display_final_instructions()
            break
        elif choice == '2':
            download_model_suite("15M")
            display_final_instructions()
            break
        elif choice == '3':
            download_model_suite("260K")
            download_model_suite("15M")
            display_final_instructions()
            break
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 4.\n")

if __name__ == "__main__":
    main() 