# OpenClip Pro - Ngrok Deployment Guide

## Prerequisites Checklist

### 1. System Requirements
- [ ] Python 3.8 or higher installed
- [ ] FFmpeg and FFprobe installed and in PATH
- [ ] At least 2GB RAM available
- [ ] Stable internet connection

### 2. Ngrok Setup
- [ ] Download ngrok from https://ngrok.com/download
- [ ] Extract and add ngrok to your system PATH
- [ ] (Optional) Sign up for ngrok account for persistent URLs
- [ ] (Optional) Get your auth token from https://dashboard.ngrok.com/auth

### 3. Python Dependencies
```bash
# Ensure all requirements are installed
pip install -r requirements.txt

# Additionally install ngrok wrapper (optional)
pip install pyngrok
```

## Quick Start Deployment

### Method 1: Using the Deployment Script
```bash
# Basic deployment (no auth)
python deploy_ngrok.py

# With ngrok auth token
python deploy_ngrok.py --auth-token YOUR_AUTH_TOKEN

# Custom port
python deploy_ngrok.py --port 8502
```

### Method 2: Manual Deployment
```bash
# Terminal 1: Start Streamlit
streamlit run openclip_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true

# Terminal 2: Start ngrok
ngrok http 8501
```

## Configuration Optimization

### 1. Streamlit Configuration (.streamlit/config.toml)
The configuration file has been optimized for ngrok deployment with:
- `headless = true` - No browser auto-open
- `enableCORS = false` - Allow cross-origin requests
- `address = "0.0.0.0"` - Listen on all interfaces
- `enableXsrfProtection = false` - Disable XSRF for tunneling

### 2. Application Settings
- Ensure `DEFAULT_TEMP_DIR` exists and has write permissions
- Configure API keys if using AI features
- Set appropriate logging levels in `config.py`

## Common Issues and Solutions

### Issue 1: "Connection Failed" or Blank Page
**Symptoms:** Ngrok URL shows but app doesn't load

**Solutions:**
1. Check if Streamlit is running locally: http://localhost:8501
2. Ensure firewall isn't blocking connections
3. Try disabling Windows Defender temporarily
4. Check ngrok dashboard for connection status

### Issue 2: "Invalid Host Header"
**Symptoms:** Error message about invalid host

**Solutions:**
1. Ensure `enableCORS = false` in config.toml
2. Add `--server.enableCORS false` to streamlit command
3. Check ngrok is using HTTP not HTTPS initially

### Issue 3: WebSocket Connection Issues
**Symptoms:** App loads but interactions don't work

**Solutions:**
1. Ensure `enableWebsocketCompression = true` in config
2. Try using ngrok with websocket support:
   ```bash
   ngrok http 8501 --scheme http --host-header localhost:8501
   ```

### Issue 4: Session State Loss
**Symptoms:** App resets frequently

**Solutions:**
1. Increase server timeout settings
2. Ensure stable internet connection
3. Consider using ngrok paid tier for stable tunnels

### Issue 5: Slow Performance
**Symptoms:** App is sluggish over ngrok

**Solutions:**
1. Optimize media file sizes
2. Enable caching in Streamlit
3. Use ngrok region closest to you:
   ```bash
   ngrok http 8501 --region us  # or eu, ap, au, sa, jp, in
   ```

## Security Considerations

### 1. Authentication
Since ngrok creates public URLs, consider:
- Using ngrok's built-in authentication:
  ```bash
  ngrok http 8501 --auth "user:password"
  ```
- Implementing app-level authentication
- Using ngrok IP restrictions (paid feature)

### 2. Data Protection
- Don't expose sensitive API keys
- Be cautious with file uploads
- Consider encryption for sensitive data
- Regularly rotate ngrok URLs

### 3. Access Control
- Monitor ngrok dashboard for connections
- Set up webhook notifications for access
- Use temporary deployments only

## Performance Optimization

### 1. Caching
Add to your Streamlit code:
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_operation():
    # Your code here
    pass
```

### 2. File Handling
- Implement file size limits
- Use streaming for large files
- Clean up temporary files regularly

### 3. Network Optimization
- Minimize API calls
- Use batch operations where possible
- Implement progress indicators

## Monitoring and Debugging

### 1. Enable Debug Logging
```python
# In config.py
LOGGING_CONFIG = {
    "level": logging.DEBUG,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}
```

### 2. Monitor Resources
```bash
# Check memory usage
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Monitor network traffic
netstat -an | findstr :8501
```

### 3. Ngrok Inspection
- Visit http://localhost:4040 for ngrok inspector
- Check request/response details
- Monitor tunnel status

## Advanced Deployment Options

### 1. Custom Domain (Paid Ngrok)
```bash
ngrok http 8501 --domain=your-custom-domain.ngrok.io
```

### 2. Multiple Tunnels
Create `ngrok.yml`:
```yaml
tunnels:
  openclip:
    proto: http
    addr: 8501
    inspect: true
  api:
    proto: http
    addr: 8000
    inspect: false
```

Run with:
```bash
ngrok start --all --config ngrok.yml
```

### 3. Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y ffmpeg
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "openclip_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Troubleshooting Checklist

If deployment fails, check:
1. [ ] All dependencies installed correctly
2. [ ] No port conflicts (8501 free)
3. [ ] Firewall rules allow connections
4. [ ] Ngrok is properly authenticated (if using auth)
5. [ ] Streamlit config file is properly formatted
6. [ ] No syntax errors in Python code
7. [ ] Sufficient system resources
8. [ ] Internet connection is stable
9. [ ] FFmpeg is accessible from PATH
10. [ ] All required directories exist with proper permissions

## Getting Help

1. Check Streamlit logs: Look for errors in console
2. Check ngrok logs: `ngrok http 8501 --log stdout`
3. Enable debug mode in both Streamlit and your app
4. Check the ngrok dashboard for tunnel status
5. Review this guide's troubleshooting section

## Contact and Support

- OpenClip Pro Issues: [GitHub Issues](https://github.com/openclip-pro/issues)
- Ngrok Documentation: https://ngrok.com/docs
- Streamlit Documentation: https://docs.streamlit.io 