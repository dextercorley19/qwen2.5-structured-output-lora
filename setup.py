#!/usr/bin/env python3
"""
Setup script for the Pitch Deck VLM Fine-tuning Pipeline.
Handles installation, configuration, and initial setup.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil


def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*60)
    print(text.center(60))
    print("="*60)


def print_step(step_num, description):
    """Print a formatted step."""
    print(f"\n[Step {step_num}] {description}")


def check_python_version():
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def install_system_dependencies():
    """Install system dependencies based on the operating system."""
    system = platform.system().lower()
    
    print(f"Detected operating system: {system}")
    
    if system == "darwin":  # macOS
        print("Installing dependencies for macOS...")
        try:
            # Check if Homebrew is installed
            subprocess.run(["brew", "--version"], check=True, capture_output=True)
            print("✅ Homebrew found")
            
            # Install poppler for PDF processing
            result = subprocess.run(["brew", "install", "poppler"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Poppler installed successfully")
            else:
                print("⚠️  Poppler installation may have failed")
                print(result.stderr)
                
        except subprocess.CalledProcessError:
            print("❌ Homebrew not found. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
            
    elif system == "linux":
        print("Installing dependencies for Linux...")
        try:
            # Try to install poppler using apt
            result = subprocess.run(["sudo", "apt-get", "update"], capture_output=True)
            if result.returncode == 0:
                result = subprocess.run(["sudo", "apt-get", "install", "-y", "poppler-utils"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print("✅ Poppler installed successfully")
                else:
                    print("⚠️  Poppler installation may have failed")
                    print(result.stderr)
            else:
                print("⚠️  Could not update package list. Please install poppler-utils manually:")
                print("   sudo apt-get install poppler-utils")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            return False
            
    elif system == "windows":
        print("⚠️  Windows detected. Please install poppler manually:")
        print("1. Download poppler from: https://github.com/oschwartz10612/poppler-windows")
        print("2. Extract to a folder (e.g., C:\\poppler)")
        print("3. Add the bin folder to your PATH environment variable")
        print("4. Restart your terminal/command prompt")
        
    else:
        print(f"⚠️  Unsupported operating system: {system}")
        print("Please install poppler manually for your system")
    
    return True


def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ pip upgraded")
        
        # Install requirements
        requirements_file = Path("requirements.txt")
        if requirements_file.exists():
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Python dependencies installed")
        else:
            print("❌ requirements.txt not found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False
    
    return True


def setup_environment_file():
    """Setup environment configuration file."""
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists() and not env_file.exists():
        shutil.copy(env_example, env_file)
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file with your API keys")
        return True
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("❌ .env.example file not found")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "processed_images",
        "training_data", 
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def verify_api_keys():
    """Check if API keys are configured."""
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = {
        "GOOGLE_API_KEY": "Gemini API",
        "LANGFUSE_PUBLIC_KEY": "Langfuse",
        "LANGFUSE_SECRET_KEY": "Langfuse"
    }
    
    missing_keys = []
    for key, service in required_keys.items():
        if not os.getenv(key):
            missing_keys.append(f"{key} ({service})")
    
    if missing_keys:
        print("⚠️  Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease edit .env file with your API keys")
        return False
    else:
        print("✅ All required API keys found")
        return True


def test_installation():
    """Test the installation by running component tests."""
    print("Testing installation...")
    
    try:
        # Run basic import tests
        import pdf2image
        import google.generativeai
        import langfuse
        import transformers
        import datasets
        print("✅ All major dependencies can be imported")
        
        # Run component tests
        result = subprocess.run([sys.executable, "test_components.py", "--component", "pdf"], 
                              capture_output=True, text=True, timeout=60)
        
        if "PDF processor initialized" in result.stdout:
            print("✅ PDF processor test passed")
        else:
            print("⚠️  PDF processor test may have failed")
            
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Test timed out")
        return False
    except Exception as e:
        print(f"⚠️  Test error: {e}")
        return False
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print_header("SETUP COMPLETE!")
    
    print("\n🎉 The Pitch Deck VLM Pipeline has been set up successfully!")
    
    print("\nNext Steps:")
    print("1. Edit the .env file with your API keys:")
    print("   - GOOGLE_API_KEY (required)")
    print("   - LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY (required)")
    print("   - HF_TOKEN (optional, for model uploads)")
    print("   - Modal tokens (optional, for cloud fine-tuning)")
    
    print("\n2. Test your setup:")
    print("   python test_components.py")
    
    print("\n3. Run a test with a few PDFs:")
    print("   python pipeline.py --max-pdfs 2")
    
    print("\n4. Run the full pipeline:")
    print("   python pipeline.py")
    
    print("\n5. Fine-tune a model (after dataset creation):")
    print("   modal run modal_finetuning.py")
    
    print("\n📚 For detailed instructions, see README.md")
    print("\n🚀 Happy fine-tuning!")


def main():
    """Main setup function."""
    print_header("PITCH DECK VLM PIPELINE SETUP")
    
    success = True
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        return False
    
    # Step 2: Install system dependencies
    print_step(2, "Installing system dependencies")
    if not install_system_dependencies():
        success = False
    
    # Step 3: Install Python dependencies
    print_step(3, "Installing Python dependencies")
    if not install_python_dependencies():
        return False
    
    # Step 4: Setup environment file
    print_step(4, "Setting up environment configuration")
    if not setup_environment_file():
        success = False
    
    # Step 5: Create directories
    print_step(5, "Creating directories")
    if not create_directories():
        success = False
    
    # Step 6: Verify API keys
    print_step(6, "Checking API keys")
    if not verify_api_keys():
        success = False
    
    # Step 7: Test installation
    print_step(7, "Testing installation")
    if not test_installation():
        success = False
    
    if success:
        print_next_steps()
    else:
        print("\n⚠️  Setup completed with some warnings.")
        print("Please review the messages above and fix any issues.")
        print("You can re-run this setup script at any time.")
    
    return success


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        sys.exit(1)
