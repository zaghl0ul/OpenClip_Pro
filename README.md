# OpenClip Pro

A powerful AI-enhanced application for visual media analysis and editing.

## Overview

OpenClip Pro is a Streamlit-based application that provides advanced tools for managing and analyzing visual media projects. It integrates with AI models (like Google's Gemini and local Llama models via Ollama) to enhance your workflow.

## Features

- Project-based organization for your media clips and assets
- AI-powered media analysis
- Easy-to-use interface with multiple view options
- API key management for various AI providers
- Customizable settings
- Integration with Ollama for local LLM inference

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/openclip-pro/openclip_pro.git
   cd openclip_pro
   ```

2. Install the required dependencies:
   ```bash
   pip install -r openclip_pro/requirements.txt
   ```

3. Install system dependencies (if required):
   ```bash
   cat openclip_pro/PACKAGES.txt | xargs apt-get install -y  # On Debian/Ubuntu
   ```

4. Install Ollama for local AI model support:
   Download and install Ollama from [https://ollama.com/download](https://ollama.com/download)

## Running the Application

Run the application using the main script:

```bash
python run_app.py
```

This will:
1. Check if the Ollama server is running and start it if needed
2. Launch the OpenClip Pro application in your default web browser

The script automatically handles Ollama server startup, so you don't need to manually start it.

Alternatively, you can run it directly with Streamlit (but you'll need to start Ollama manually):

```bash
streamlit run openclip_pro/openclip_app.py
```

## Testing Your Setup

Before running the full application, you can verify that your API connections are working correctly:

```bash
python test_apis.py
```

This test script will:
1. Check if Ollama is installed and running (and start it if needed)
2. Test connectivity to the Ollama API and list available models
3. Test your Gemini API key and connectivity
4. Provide a summary of test results

If any tests fail, the script will provide guidance on how to fix the issues.

## Using Ollama Models

OpenClip Pro supports local LLM inference using Ollama. The application can use any model you have installed in Ollama.

### Installing LLaVA (Vision Model)

To use the LLaVA vision model for image analysis:

```bash
ollama pull llava
```

### Installing Other Models

You can install other Ollama models as needed:

```bash
ollama pull mistral     # Install Mistral model
ollama pull llama3      # Install Llama 3 model
```

## Configuration

### API Keys

API keys for AI services are stored in `api_keys.json` in the root directory. 
The file should have the following format:

```json
{
  "Gemini": "your-gemini-api-key",
  "OtherProvider": "other-provider-api-key"
}
```

You can also manage API keys directly through the application's interface.

### Ollama Configuration

By default, the application connects to Ollama at http://localhost:11434. You can override this by setting the `OLLAMA_HOST` environment variable:

```bash
# Example: Running Ollama on a different machine
export OLLAMA_HOST=http://192.168.1.100:11434
```

### Application Settings

Application settings are stored in the database and can be configured through the Settings page in the application.

## Project Structure

```
openclip_pro/
├── ai/                  # AI integration modules
├── ui/                  # User interface components
│   └── components/      # Reusable UI components
├── openclip_app.py      # Main application file
├── config.py            # Configuration settings
├── database.py          # Database operations
└── requirements.txt     # Python dependencies
```

## License

Copyright © 2024 OpenClip Pro

## Contact

For support or inquiries, visit [the documentation](https://github.com/openclip-pro/documentation). 
