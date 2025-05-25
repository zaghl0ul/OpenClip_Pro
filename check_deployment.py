#!/usr/bin/env python3
"""
OpenClip Pro - Pre-deployment Check Script
This script validates the environment and configuration before deployment.
"""

import os
import sys
import subprocess
import json
import platform
import socket
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class DeploymentChecker:
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Gather system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': sys.version,
            'hostname': socket.gethostname(),
            'working_directory': os.getcwd()
        }
    
    def check_python_version(self) -> bool:
        """Check if Python version is 3.8 or higher."""
        logger.info("Checking Python version...")
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.issues.append(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        self.info.append(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_required_files(self) -> bool:
        """Check if all required files exist."""
        logger.info("Checking required files...")
        required_files = [
            'openclip_app.py',
            'requirements.txt',
            'config.py',
            'database.py',
            'media_utils.py',
            'ui_components.py'
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            self.issues.append(f"Missing required files: {', '.join(missing)}")
            return False
        
        self.info.append("✓ All required files present")
        return True
    
    def check_directories(self) -> bool:
        """Check and create necessary directories."""
        logger.info("Checking directories...")
        directories = [
            '.streamlit',
            'tmp',
            'ai',
            'ui'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    self.info.append(f"✓ Created directory: {directory}")
                except Exception as e:
                    self.issues.append(f"Failed to create directory {directory}: {e}")
                    return False
            else:
                self.info.append(f"✓ Directory exists: {directory}")
        
        return True
    
    def check_streamlit_config(self) -> bool:
        """Check Streamlit configuration."""
        logger.info("Checking Streamlit configuration...")
        config_path = '.streamlit/config.toml'
        
        if not os.path.exists(config_path):
            self.warnings.append("Streamlit config not found. Default config will be used.")
            return True
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                
            # Check for ngrok-friendly settings
            required_settings = [
                'headless = true',
                'enableCORS = false',
                'address = "0.0.0.0"'
            ]
            
            missing_settings = []
            for setting in required_settings:
                if setting not in content:
                    missing_settings.append(setting)
            
            if missing_settings:
                self.warnings.append(f"Recommended settings missing in config.toml: {missing_settings}")
            else:
                self.info.append("✓ Streamlit config properly configured for ngrok")
                
        except Exception as e:
            self.warnings.append(f"Could not read Streamlit config: {e}")
        
        return True
    
    def check_dependencies(self) -> bool:
        """Check if all Python dependencies are installed."""
        logger.info("Checking Python dependencies...")
        
        try:
            # Check core dependencies
            core_deps = ['streamlit', 'requests', 'pillow', 'opencv-python']
            missing_deps = []
            
            for dep in core_deps:
                try:
                    __import__(dep.replace('-', '_'))
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                self.issues.append(f"Missing core dependencies: {', '.join(missing_deps)}")
                self.info.append("Run: pip install -r requirements.txt")
                return False
            
            self.info.append("✓ Core Python dependencies installed")
            return True
            
        except Exception as e:
            self.warnings.append(f"Could not check all dependencies: {e}")
            return True
    
    def check_ffmpeg(self) -> bool:
        """Check if ffmpeg is installed and accessible."""
        logger.info("Checking ffmpeg installation...")
        
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                self.info.append(f"✓ FFmpeg found: {version_line}")
                return True
            else:
                self.issues.append("FFmpeg check failed")
                return False
        except FileNotFoundError:
            self.issues.append("FFmpeg not found in PATH")
            self.info.append("Install ffmpeg from: https://ffmpeg.org/download.html")
            return False
        except Exception as e:
            self.issues.append(f"Error checking ffmpeg: {e}")
            return False
    
    def check_ngrok(self) -> bool:
        """Check if ngrok is installed."""
        logger.info("Checking ngrok installation...")
        
        try:
            result = subprocess.run(['ngrok', 'version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.info.append(f"✓ Ngrok found: {result.stdout.strip()}")
                return True
            else:
                self.warnings.append("Ngrok check returned non-zero exit code")
                return True
        except FileNotFoundError:
            self.warnings.append("Ngrok not found in PATH")
            self.info.append("Install ngrok from: https://ngrok.com/download")
            return True  # Not critical, warning only
        except Exception as e:
            self.warnings.append(f"Error checking ngrok: {e}")
            return True
    
    def check_port_availability(self, port: int = 8501) -> bool:
        """Check if the default port is available."""
        logger.info(f"Checking port {port} availability...")
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('', port))
            sock.close()
            self.info.append(f"✓ Port {port} is available")
            return True
        except socket.error:
            self.warnings.append(f"Port {port} is already in use")
            self.info.append("The deployment script will handle this")
            return True
    
    def check_api_keys(self) -> bool:
        """Check if API keys file exists (optional)."""
        logger.info("Checking API keys configuration...")
        
        api_key_file = 'api_keys.json'
        if os.path.exists(api_key_file):
            try:
                with open(api_key_file, 'r') as f:
                    data = json.load(f)
                self.info.append("✓ API keys file found")
                
                # Check if file has proper permissions (Windows)
                if platform.system() == 'Windows':
                    self.warnings.append("Remember to protect your API keys file")
                    
            except Exception as e:
                self.warnings.append(f"Could not read API keys file: {e}")
        else:
            self.info.append("ℹ API keys file not found (optional)")
        
        return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space."""
        logger.info("Checking disk space...")
        
        try:
            import shutil
            stat = shutil.disk_usage(os.getcwd())
            free_gb = stat.free / (1024 ** 3)
            
            if free_gb < 1:
                self.warnings.append(f"Low disk space: {free_gb:.2f} GB free")
            else:
                self.info.append(f"✓ Disk space: {free_gb:.2f} GB free")
                
        except Exception as e:
            self.warnings.append(f"Could not check disk space: {e}")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run all deployment checks."""
        logger.info("Starting pre-deployment checks...")
        logger.info(f"System: {self.system_info['platform']} {self.system_info['platform_version']}")
        
        checks = [
            self.check_python_version,
            self.check_required_files,
            self.check_directories,
            self.check_streamlit_config,
            self.check_dependencies,
            self.check_ffmpeg,
            self.check_ngrok,
            self.check_port_availability,
            self.check_api_keys,
            self.check_disk_space
        ]
        
        all_passed = True
        for check in checks:
            if not check():
                all_passed = False
        
        return all_passed and len(self.issues) == 0
    
    def print_report(self):
        """Print the deployment check report."""
        print("\n" + "="*60)
        print("🔍 OpenClip Pro - Deployment Check Report")
        print("="*60)
        
        if self.issues:
            print("\n❌ CRITICAL ISSUES (must fix before deployment):")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS (review but not critical):")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if self.info:
            print("\nℹ️  INFORMATION:")
            for info in self.info:
                print(f"   {info}")
        
        print("\n" + "="*60)
        
        if self.issues:
            print("❌ Deployment check FAILED - Please fix critical issues")
            print("="*60)
            return False
        else:
            print("✅ Deployment check PASSED - Ready for ngrok deployment!")
            print("\nNext steps:")
            print("1. Run: python deploy_ngrok.py")
            print("   OR")
            print("2. Run: deploy_ngrok.bat (on Windows)")
            print("="*60)
            return True

def main():
    checker = DeploymentChecker()
    checker.run_all_checks()
    success = checker.print_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 