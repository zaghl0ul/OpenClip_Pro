#!/usr/bin/env python3
"""
OpenClip Pro - Ngrok Deployment Script
This script handles the deployment of OpenClip Pro to ngrok for public access.
"""

import os
import sys
import subprocess
import logging
import signal
import time
import json
import argparse
import platform
from typing import Optional, Tuple
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class NgrokDeployment:
    def __init__(self, port: int = 8501, auth_token: Optional[str] = None):
        self.port = port
        self.auth_token = auth_token
        self.streamlit_process = None
        self.ngrok_process = None
        
    def get_python_command(self) -> list:
        """Get the appropriate Python command for the system."""
        if platform.system() == 'Windows':
            # Try py launcher first on Windows
            try:
                subprocess.run(['py', '-3', '--version'], capture_output=True, check=True)
                return ['py', '-3']
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Fall back to python3 or python
        for cmd in ['python3', 'python']:
            try:
                subprocess.run([cmd, '--version'], capture_output=True, check=True)
                return [cmd]
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        raise RuntimeError("Python interpreter not found")
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        logger.info("Checking dependencies...")
        
        # Check for Python packages
        required_packages = ['streamlit', 'requests']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing Python packages: {', '.join(missing_packages)}")
            logger.info("Install them with: pip install " + ' '.join(missing_packages))
            return False
            
        # Check for ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            logger.info("✓ ffmpeg found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ ffmpeg not found. Please install ffmpeg and add it to PATH")
            return False
            
        # Check for ngrok
        try:
            subprocess.run(['ngrok', 'version'], capture_output=True, check=True)
            logger.info("✓ ngrok found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("✗ ngrok not found. Please install ngrok from https://ngrok.com/")
            return False
            
        return True
    
    def setup_ngrok_auth(self) -> bool:
        """Setup ngrok authentication if token is provided."""
        if self.auth_token:
            logger.info("Setting up ngrok authentication...")
            try:
                subprocess.run(['ngrok', 'config', 'add-authtoken', self.auth_token], 
                             check=True, capture_output=True)
                logger.info("✓ Ngrok authentication configured")
                return True
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to set ngrok auth token: {e}")
                return False
        return True
    
    def start_streamlit(self) -> bool:
        """Start the Streamlit application."""
        logger.info(f"Starting Streamlit on port {self.port}...")
        
        # Set environment variables for better performance
        env = os.environ.copy()
        env['STREAMLIT_SERVER_HEADLESS'] = 'true'
        env['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
        env['STREAMLIT_SERVER_PORT'] = str(self.port)
        env['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        try:
            # Get the appropriate Python command
            python_cmd = self.get_python_command()
            
            # Build the command
            cmd = python_cmd + ['-m', 'streamlit', 'run', 'openclip_app.py', 
                               '--server.port', str(self.port),
                               '--server.address', '0.0.0.0',
                               '--server.headless', 'true']
            
            self.streamlit_process = subprocess.Popen(cmd, env=env)
            
            # Wait for Streamlit to start
            time.sleep(5)
            
            # Check if Streamlit is running
            try:
                response = requests.get(f'http://localhost:{self.port}', timeout=5)
                if response.status_code == 200:
                    logger.info("✓ Streamlit is running")
                    return True
            except requests.exceptions.RequestException:
                pass
                
            logger.error("Streamlit failed to start properly")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start Streamlit: {e}")
            return False
    
    def start_ngrok(self) -> Optional[str]:
        """Start ngrok tunnel and return the public URL."""
        logger.info("Starting ngrok tunnel...")
        
        try:
            # Start ngrok with HTTP tunnel
            self.ngrok_process = subprocess.Popen(
                ['ngrok', 'http', str(self.port), '--log=stdout'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for ngrok to establish connection
            time.sleep(3)
            
            # Get tunnel information from ngrok API
            try:
                response = requests.get('http://localhost:4040/api/tunnels', timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data['tunnels']:
                        public_url = data['tunnels'][0]['public_url']
                        logger.info(f"✓ Ngrok tunnel established: {public_url}")
                        return public_url
            except requests.exceptions.RequestException as e:
                logger.warning(f"Could not get tunnel info from API: {e}")
                
            logger.error("Failed to establish ngrok tunnel")
            return None
            
        except Exception as e:
            logger.error(f"Failed to start ngrok: {e}")
            return None
    
    def health_check(self) -> Tuple[bool, bool]:
        """Check if both Streamlit and ngrok are running."""
        streamlit_healthy = False
        ngrok_healthy = False
        
        # Check Streamlit
        if self.streamlit_process and self.streamlit_process.poll() is None:
            try:
                response = requests.get(f'http://localhost:{self.port}', timeout=2)
                streamlit_healthy = response.status_code == 200
            except:
                pass
                
        # Check ngrok
        if self.ngrok_process and self.ngrok_process.poll() is None:
            try:
                response = requests.get('http://localhost:4040/api/tunnels', timeout=2)
                ngrok_healthy = response.status_code == 200
            except:
                pass
                
        return streamlit_healthy, ngrok_healthy
    
    def deploy(self) -> bool:
        """Deploy the application with ngrok."""
        # Check dependencies
        if not self.check_dependencies():
            return False
            
        # Setup ngrok auth if provided
        if not self.setup_ngrok_auth():
            return False
            
        # Start Streamlit
        if not self.start_streamlit():
            return False
            
        # Start ngrok
        public_url = self.start_ngrok()
        if not public_url:
            self.cleanup()
            return False
            
        # Print deployment information
        print("\n" + "="*60)
        print("🎬 OpenClip Pro is now deployed!")
        print("="*60)
        print(f"📍 Local URL: http://localhost:{self.port}")
        print(f"🌐 Public URL: {public_url}")
        print("="*60)
        print("⚠️  Security Notice:")
        print("   - This URL is publicly accessible")
        print("   - Anyone with the URL can access your application")
        print("   - Be cautious about sensitive data")
        print("="*60)
        print("\nPress Ctrl+C to stop the deployment")
        
        # Monitor the deployment
        try:
            while True:
                streamlit_ok, ngrok_ok = self.health_check()
                
                if not streamlit_ok:
                    logger.error("Streamlit is not responding!")
                    break
                    
                if not ngrok_ok:
                    logger.error("Ngrok tunnel is down!")
                    break
                    
                time.sleep(10)  # Check every 10 seconds
                
        except KeyboardInterrupt:
            logger.info("\nShutting down deployment...")
            
        self.cleanup()
        return True
    
    def cleanup(self):
        """Clean up processes."""
        if self.streamlit_process:
            logger.info("Stopping Streamlit...")
            self.streamlit_process.terminate()
            try:
                self.streamlit_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.streamlit_process.kill()
                
        if self.ngrok_process:
            logger.info("Stopping ngrok...")
            self.ngrok_process.terminate()
            try:
                self.ngrok_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ngrok_process.kill()
                
        logger.info("Cleanup complete")

def main():
    parser = argparse.ArgumentParser(
        description='Deploy OpenClip Pro with ngrok'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=8501,
        help='Port to run Streamlit on (default: 8501)'
    )
    parser.add_argument(
        '--auth-token',
        type=str,
        help='Ngrok authentication token (optional)'
    )
    
    args = parser.parse_args()
    
    # Create deployment instance
    deployment = NgrokDeployment(
        port=args.port,
        auth_token=args.auth_token
    )
    
    # Handle signals for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal...")
        deployment.cleanup()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform != 'win32':
        signal.signal(signal.SIGTERM, signal_handler)
    
    # Deploy the application
    success = deployment.deploy()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 