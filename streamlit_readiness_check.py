#!/usr/bin/env python3
"""
Streamlit Deployment Readiness Checker
Validates that all required files and configurations are in place
"""
import sys
from pathlib import Path
from typing import List, Tuple

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_file_exists(filepath: str) -> bool:
    """Check if file exists"""
    return Path(filepath).exists()


def check_file_not_empty(filepath: str) -> bool:
    """Check if file exists and is not empty"""
    path = Path(filepath)
    return path.exists() and path.stat().st_size > 0


def check_model_size(filepath: str, max_mb: int = 100) -> Tuple[bool, str]:
    """Check if model file size is appropriate"""
    path = Path(filepath)
    if not path.exists():
        return False, "Model file not found"
    
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"Model is {size_mb:.1f}MB (> {max_mb}MB). Use Git LFS or external hosting."
    
    return True, f"Model size: {size_mb:.1f}MB (OK)"


def check_requirements() -> bool:
    """Check if requirements-streamlit.txt has essential packages"""
    req_file = Path("requirements-streamlit.txt")
    if not req_file.exists():
        return False
    
    content = req_file.read_text()
    essential = ['streamlit', 'torch', 'torchvision', 'Pillow', 'opencv']
    
    missing = [pkg for pkg in essential if pkg.lower() not in content.lower()]
    
    if missing:
        print(f"  {RED}✗{RESET} Missing packages: {', '.join(missing)}")
        return False
    
    return True


def check_gitignore() -> bool:
    """Check if .gitignore includes secrets.toml"""
    gitignore = Path(".gitignore")
    if not gitignore.exists():
        return False
    
    content = gitignore.read_text()
    return "secrets.toml" in content


def run_checks() -> Tuple[int, int]:
    """Run all checks and return (passed, total)"""
    checks = [
        ("Entry Point", "streamlit_app.py", check_file_not_empty),
        ("Streamlit Requirements", "requirements-streamlit.txt", check_file_not_empty),
        ("System Packages", "packages.txt", check_file_exists),
        ("Streamlit Config", ".streamlit/config.toml", check_file_exists),
        ("Git Attributes", ".gitattributes", check_file_exists),
        ("Secrets Template", ".streamlit/secrets.toml.example", check_file_exists),
        ("Deployment Guide", "STREAMLIT_DEPLOYMENT.md", check_file_exists),
        ("Quick Start", "STREAMLIT_QUICK_START.md", check_file_exists),
    ]
    
    passed = 0
    total = len(checks)
    
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Streamlit Deployment Readiness Check{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    print(f"{YELLOW}Required Files:{RESET}")
    for name, filepath, check_func in checks:
        result = check_func(filepath)
        status = f"{GREEN}✓{RESET}" if result else f"{RED}✗{RESET}"
        print(f"  {status} {name}: {filepath}")
        if result:
            passed += 1
    
    print(f"\n{YELLOW}Content Validation:{RESET}")
    
    # Check requirements content
    if check_requirements():
        print(f"  {GREEN}✓{RESET} Essential packages in requirements-streamlit.txt")
        passed += 1
    else:
        print(f"  {RED}✗{RESET} Missing essential packages in requirements-streamlit.txt")
    total += 1
    
    # Check gitignore
    if check_gitignore():
        print(f"  {GREEN}✓{RESET} .gitignore includes secrets.toml")
        passed += 1
    else:
        print(f"  {RED}✗{RESET} .gitignore should include secrets.toml")
    total += 1
    
    # Check model file
    model_path = "models/demo_run/best_model.pth"
    if Path(model_path).exists():
        success, msg = check_model_size(model_path)
        status = f"{GREEN}✓{RESET}" if success else f"{YELLOW}⚠{RESET}"
        print(f"  {status} {msg}")
        if success:
            passed += 1
    else:
        print(f"  {YELLOW}⚠{RESET} Model file not found (will need download URL)")
    total += 1
    
    print(f"\n{YELLOW}Optional Enhancements:{RESET}")
    
    # Check optional files
    optional = [
        ("Model Downloader", "src/utils/model_downloader.py"),
        ("Git LFS configured", ".gitattributes"),
        ("Secrets file", ".streamlit/secrets.toml"),
    ]
    
    for name, filepath in optional:
        exists = check_file_exists(filepath)
        status = f"{GREEN}✓{RESET}" if exists else f"{YELLOW}○{RESET}"
        print(f"  {status} {name}")
    
    return passed, total


def print_summary(passed: int, total: int):
    """Print summary and recommendations"""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}  Summary{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    percentage = (passed / total) * 100
    
    if percentage == 100:
        status_color = GREEN
        status = "✓ READY FOR DEPLOYMENT"
    elif percentage >= 80:
        status_color = YELLOW
        status = "⚠ MOSTLY READY (minor issues)"
    else:
        status_color = RED
        status = "✗ NOT READY (missing requirements)"
    
    print(f"  {status_color}{status}{RESET}")
    print(f"  Passed: {passed}/{total} checks ({percentage:.0f}%)\n")
    
    if percentage < 100:
        print(f"{YELLOW}Recommendations:{RESET}")
        
        if not check_file_exists("streamlit_app.py"):
            print("  • Create streamlit_app.py as entry point")
        
        if not check_file_exists("requirements-streamlit.txt"):
            print("  • Create requirements-streamlit.txt with dependencies")
        
        if not check_file_exists("packages.txt"):
            print("  • Create packages.txt for system dependencies")
        
        model_path = "models/demo_run/best_model.pth"
        if Path(model_path).exists():
            size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            if size_mb > 100:
                print(f"  • Model is {size_mb:.1f}MB. Consider:")
                print(f"    - Setup Git LFS: git lfs track '*.pth'")
                print(f"    - Or host externally (Hugging Face, GDrive)")
                print(f"    - Or use model quantization to reduce size")
        
        if not check_gitignore():
            print("  • Add 'secrets.toml' to .gitignore")
        
        print()
    
    print(f"{YELLOW}Next Steps:{RESET}")
    if percentage >= 80:
        print("  1. Test locally: streamlit run streamlit_app.py")
        print("  2. Push to GitHub: git push origin main")
        print("  3. Deploy on Streamlit Cloud: share.streamlit.io")
        print("  4. Configure secrets in Streamlit Cloud UI")
    else:
        print("  1. Fix missing requirements above")
        print("  2. Re-run this check: python streamlit_readiness_check.py")
        print("  3. See STREAMLIT_DEPLOYMENT.md for detailed guide")
    
    print(f"\n{BLUE}{'='*60}{RESET}\n")


def main():
    """Main entry point"""
    # Change to project root
    script_dir = Path(__file__).parent
    if (script_dir / "src").exists():
        # Ensure we run from project root for relative paths
        import os
        os.chdir(script_dir)
    elif (script_dir.parent / "src").exists():
        import os
        os.chdir(script_dir.parent)
    
    passed, total = run_checks()
    print_summary(passed, total)
    
    # Exit code: 0 if all passed, 1 otherwise
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
