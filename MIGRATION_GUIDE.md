# Migration Guide - OpenClip Pro Refactoring

This guide helps you migrate from the old codebase to the refactored architecture.

## Overview of Changes

### 1. New Module Structure
```
OpenClip_Pro/
├── ai/
│   ├── base.py              # Base classes and interfaces
│   ├── error_handling.py    # Centralized error handling
│   ├── model_manager.py     # Simplified AI model manager
│   ├── provider_factory.py  # Provider management
│   └── providers/           # Provider implementations
│       ├── openai_provider.py
│       ├── anthropic_provider.py
│       ├── google_provider.py
│       └── ollama_provider.py
├── models/
│   └── data_models.py       # Type-safe data models
├── utils/
│   ├── config_manager.py    # Centralized configuration
│   ├── logging_config.py    # Logging configuration
│   └── path_manager.py      # Path management
```

### 2. Key Benefits
- **Type Safety**: Strong typing with dataclasses and type hints
- **Error Handling**: Consistent error handling with decorators
- **Configuration**: Centralized configuration management
- **Logging**: Unified logging with proper formatting
- **Path Management**: Safe and consistent path operations
- **Provider Abstraction**: Clean interface for AI providers

## Migration Steps

### Step 1: Update Imports

**Old:**
```python
from ai.ai_models import AIAnalysisModule, APIKeyManager, ModelRegistry
from media_utils import create_project_directories
from database import save_project, load_project
import logging
```

**New:**
```python
from ai.model_manager import AIModelManager
from ai.base import ProviderType
from utils import get_config, setup_logging, get_logger, get_path_manager
from models import ProjectData, ClipData
```

### Step 2: Initialize Components

**Old:**
```python
# Scattered initialization
logger = logging.getLogger(__name__)
ai_module = AIAnalysisModule()
```

**New:**
```python
# Centralized initialization
from utils import setup_logging, get_logger, get_config
from ai.model_manager import AIModelManager

# Setup logging once at app start
setup_logging(
    log_file="logs/app.log",
    console=True,
    use_colors=True,
    detailed=config.debug_mode
)

# Get logger for module
logger = get_logger(__name__)

# Get configuration
config = get_config()

# Initialize AI manager (with proper dependencies)
ai_manager = AIModelManager(key_manager, model_registry)
```

### Step 3: Path Management

**Old:**
```python
import os
base_dir = os.path.join(os.path.expanduser("~"), ".openclip", project_id)
os.makedirs(base_dir, exist_ok=True)
clips_dir = os.path.join(base_dir, "clips")
os.makedirs(clips_dir, exist_ok=True)
```

**New:**
```python
from utils import get_path_manager

path_manager = get_path_manager()
project_paths = path_manager.create_project_directories(project_id)
clips_dir = project_paths["clips"]
```

### Step 4: Error Handling

**Old:**
```python
try:
    result = some_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}")
    return None
```

**New:**
```python
from ai.error_handling import handle_errors, error_context

# Using decorator
@handle_errors(default_return=None, error_prefix="Operation")
def some_operation():
    # operation code
    pass

# Using context manager
with error_context("Processing data", default_return=[]):
    result = process_data()
```

### Step 5: AI Analysis

**Old:**
```python
response_text = await ai_module.model_manager.analyze_with_model(
    provider="OpenAI",
    model_name="gpt-4o",
    prompt=prompt,
    images=images,
    format_type="json"
)

if response_text.startswith("Error:"):
    # Handle error
```

**New:**
```python
from ai.base import AnalysisResponse

response = await ai_manager.analyze(
    provider="openai",  # or ProviderType.OPENAI
    model_name="gpt-4o",
    prompt=prompt,
    images=images,
    format_type="json"
)

if response.is_error:
    error_msg = response.get_error_message()
    # Handle error
else:
    content = response.content
```

### Step 6: Data Models

**Old:**
```python
# Using dictionaries
clip_data = {
    "id": str(uuid.uuid4()),
    "start": 0.0,
    "end": 60.0,
    "score": 85,
    "tag": "Highlight",
    # ...
}

project_data = {
    "id": project_id,
    "name": project_name,
    "clips": clips,
    # ...
}
```

**New:**
```python
from models import ClipData, ProjectData

# Using type-safe dataclasses
clip = ClipData(
    start=0.0,
    end=60.0,
    score=85,
    tag="Highlight"
)

project = ProjectData(
    name=project_name,
    clips=[clip]
)

# Convert to dict for storage
clip_dict = clip.to_dict()
project_dict = project.to_dict()

# Load from dict
clip = ClipData.from_dict(clip_dict)
project = ProjectData.from_dict(project_dict)
```

### Step 7: Configuration

**Old:**
```python
# Hardcoded values scattered throughout
DEFAULT_CLIP_LENGTH = 60
DEFAULT_TEMPERATURE = 0.5
API_TIMEOUT = 120
```

**New:**
```python
from utils import get_config

config = get_config()

# Access configuration values
clip_length = config.default_clip_length
temperature = config.default_temperature
timeout = config.default_api_timeout

# Update configuration
config_manager = get_config_manager()
config_manager.set("default_clip_length", 90)
config_manager.save_config()
```

### Step 8: Provider-Specific Code

**Old:**
```python
# Large if/elif blocks for each provider
if provider_lower == "openai":
    # OpenAI specific code
elif provider_lower == "anthropic":
    # Anthropic specific code
elif provider_lower == "google":
    # Google specific code
```

**New:**
```python
# Providers implement common interface
provider = await provider_factory.get_provider(ProviderType.OPENAI)
response = await provider.analyze(request)
```

## Common Patterns

### Async/Await Consistency

**Old:**
```python
# Mixed async/sync with ThreadPoolExecutor
executor = ThreadPoolExecutor()
result = await loop.run_in_executor(executor, sync_function)
```

**New:**
```python
# Consistent async-first approach
result = await async_function()
```

### Validation

**Old:**
```python
# Manual validation
if not prompt or len(prompt) == 0:
    raise ValueError("Prompt cannot be empty")
if temperature < 0 or temperature > 2:
    raise ValueError("Temperature must be between 0 and 2")
```

**New:**
```python
from ai.error_handling import validate_input

validate_input(prompt, str, "prompt", min_length=1)
validate_input(temperature, float, "temperature", min_value=0.0, max_value=2.0)
```

### Logging

**Old:**
```python
print(f"DEBUG: {message}")
logger.debug(message)
logger.error(f"Error: {e}", exc_info=True)
```

**New:**
```python
logger = get_logger(__name__)
logger.debug(message)  # No print statements
logger.error(f"Error context: {e}", exc_info=True)  # Consistent format
```

## Testing the Migration

1. **Unit Tests**: Update tests to use new interfaces
2. **Integration Tests**: Test provider implementations
3. **Type Checking**: Run mypy to catch type errors
4. **Error Scenarios**: Test error handling paths

## Rollback Plan

If issues arise:
1. Keep old `ai_models.py` as `ai_models_legacy.py`
2. Use feature flags to switch between implementations
3. Gradually migrate one component at a time

## Support

For questions or issues during migration:
1. Check the refactoring analysis document
2. Review the new module documentation
3. Test in a development environment first 