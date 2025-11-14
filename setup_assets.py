"""
Setup script to download chess piece images
Run this script to automatically download chess piece images
"""
import os
import sys

def create_assets_folder():
    """Create assets folder if it doesn't exist"""
    assets_path = 'assets/img/chess_pieces'
    if not os.path.exists(assets_path):
        os.makedirs(assets_path)
        print(f"✓ Created '{assets_path}' folder")
    else:
        print(f"✓ '{assets_path}' folder already exists")

def check_images():
    """Check if all required images exist"""
    required_images = [
        'white_pawn.png', 'white_knight.png', 'white_bishop.png',
        'white_rook.png', 'white_queen.png', 'white_king.png',
        'black_pawn.png', 'black_knight.png', 'black_bishop.png',
        'black_rook.png', 'black_queen.png', 'black_king.png'
    ]
    
    assets_path = 'assets/img/chess_pieces'
    missing = []
    for img in required_images:
        if not os.path.exists(os.path.join(assets_path, img)):
            missing.append(img)
    
    if missing:
        print(f"\n❌ Missing {len(missing)} images:")
        for img in missing:
            print(f"   - {img}")
        return False
    else:
        print("\n✓ All chess piece images found!")
        return True

def download_images():
    """Download chess piece images from Wikimedia Commons"""
    try:
        import urllib.request
        
        print("\nDownloading chess piece images from Wikimedia Commons...")
        
        # URLs for chess pieces from Wikimedia Commons (public domain)
        base_url = "https://upload.wikimedia.org/wikipedia/commons/"
        
        pieces = {
            'white_pawn.png': '4/45/Chess_plt45.svg.png',
            'white_knight.png': '7/70/Chess_nlt45.svg.png',
            'white_bishop.png': 'b/b1/Chess_blt45.svg.png',
            'white_rook.png': '7/72/Chess_rlt45.svg.png',
            'white_queen.png': '1/15/Chess_qlt45.svg.png',
            'white_king.png': '4/42/Chess_klt45.svg.png',
            'black_pawn.png': 'c/c7/Chess_pdt45.svg.png',
            'black_knight.png': 'e/ef/Chess_ndt45.svg.png',
            'black_bishop.png': '9/98/Chess_bdt45.svg.png',
            'black_rook.png': 'f/ff/Chess_rdt45.svg.png',
            'black_queen.png': '4/47/Chess_qdt45.svg.png',
            'black_king.png': 'f/f0/Chess_kdt45.svg.png'
        }
        
        assets_path = 'assets/img/chess_pieces'
        for filename, url_path in pieces.items():
            url = base_url + url_path
            filepath = os.path.join(assets_path, filename)
            
            try:
                print(f"  Downloading {filename}...", end='')
                urllib.request.urlretrieve(url, filepath)
                print(" ✓")
            except Exception as e:
                print(f" ✗ Failed: {e}")
        
        print("\n✓ Download complete!")
        return True
        
    except ImportError:
        print("\n❌ urllib not available. Cannot download images automatically.")
        return False

def print_manual_instructions():
    """Print instructions for manual image setup"""
    print("\n" + "="*70)
    print("MANUAL SETUP INSTRUCTIONS")
    print("="*70)
    print("\nYou need 12 chess piece images in the 'assets' folder:")
    print("\nWhite pieces:")
    print("  - white_pawn.png")
    print("  - white_knight.png")
    print("  - white_bishop.png")
    print("  - white_rook.png")
    print("  - white_queen.png")
    print("  - white_king.png")
    print("\nBlack pieces:")
    print("  - black_pawn.png")
    print("  - black_knight.png")
    print("  - black_bishop.png")
    print("  - black_rook.png")
    print("  - black_queen.png")
    print("  - black_king.png")
    print("\nWhere to find chess piece images:")
    print("  1. Wikimedia Commons (public domain):")
    print("     https://commons.wikimedia.org/wiki/Category:SVG_chess_pieces")
    print("  2. OpenGameArt.org (free/open source)")
    print("  3. Create your own using image editing software")
    print("\nRecommended image size: 256x256 pixels or larger (PNG format)")
    print("\nAfter adding the images, run this script again to verify.")
    print("="*70)

def main():
    print("="*70)
    print("Chess Tournament - Assets Setup")
    print("="*70)
    
    # Create assets folder
    create_assets_folder()
    
    # Check if images already exist
    if check_images():
        print("\n✓ Setup complete! You can run the chess tournament program.")
        return
    
    print("\nOptions:")
    print("1. Automatically download images (recommended)")
    print("2. Setup manually")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        if download_images():
            if check_images():
                print("\n✓ Setup complete! You can now run the chess tournament program.")
            else:
                print("\n⚠ Some images may not have downloaded correctly.")
                print("The program will use Unicode text symbols as fallback.")
    elif choice == '2':
        print_manual_instructions()
    else:
        print("\nExiting. Run this script again when you're ready to setup assets.")

if __name__ == "__main__":
    main()

