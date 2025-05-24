import asyncio
import base64
import binascii
import concurrent.futures  # Used for running async code in sync contexts
import json
import logging
import os  # Import for file operations
import time  # Used in UI feedback
import tempfile
from typing import Any, Dict, List, Optional, Union
import requests
import streamlit as st

# Setup logging first
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import streamlit early for error handling
import streamlit as st

# Import Fernet for encryption
try:
    from cryptography.fernet import Fernet  # For API key encryption
except ImportError:
    logger.error("cryptography package not installed. API key encryption will not work.")
    Fernet = None  # Set to None so we can check if it's available


# --- Utility Functions ---
def debug_print(message: str):
    """Debug print function to replace missing debug_print calls."""
    logger.debug(message)


# --- Configuration ---
# Assuming config.py defines API_KEY_FILE and other relevant paths/settings
try:
    from config import API_KEY_FILE
except ImportError:
    logger.error(
        "config.py not found. Please ensure it exists and defines API_KEY_FILE."
    )
    # Provide a default fallback path or raise error? For now, use a default.
    API_KEY_FILE = "data/api_keys.enc"
    logger.warning(f"Using default API_KEY_FILE path: {API_KEY_FILE}")

# Assuming media_utils.py defines optimize_and_encode_image
try:
    from media_utils import optimize_and_encode_image
except ImportError:
    logger.error("media_utils.py not found or optimize_and_encode_image is missing.")

    # Define a dummy function to prevent NameErrors later if media_utils is missing
    def optimize_and_encode_image(path, quality=85, max_resolution=720):
        logger.warning(
            "Using dummy optimize_and_encode_image function. Image processing may fail."
        )
        # Attempt a basic base64 encoding as a fallback
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Dummy image encoding failed for {path}: {e}")
            return None


# Assuming database_utils.py defines update_clip_data (placeholder for now)
# from database_utils import update_clip_data
def update_clip_data(clip_id: str, data: Dict) -> bool:
    """Placeholder: Updates clip data in the database."""
    logger.debug(f"Placeholder: Updating DB for clip {clip_id} with keys {data.keys()}")
    # In a real scenario, this would interact with the database
    # For this refactor, assume it works.
    return True


# --- AI Provider Client Imports ---
# Use try-except for optional dependencies and log warnings if not found.
# Prefer async clients for asyncio-based applications.

# OpenAI
try:
    from openai import APIError, AsyncOpenAI, Timeout as OpenAITimeout
except ImportError:
    AsyncOpenAI = None
    APIError = None
    OpenAITimeout = None
    logger.warning("openai package not found. OpenAI models will not be available.")

# Anthropic
try:
    import anthropic
    from anthropic import APIError as AnthropicAPIError, Timeout as AnthropicTimeout
except ImportError:
    anthropic = None
    AnthropicAPIError = None
    AnthropicTimeout = None
    logger.warning("anthropic package not found. Claude models will not be available.")

# Google Generative AI
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    from google.api_core import exceptions as GoogleAPIErrors
    # Verify the import worked by checking for key attributes
    if not hasattr(genai, 'configure'):
        raise ImportError("google.generativeai.configure not found")
    if not hasattr(genai, 'GenerativeModel'):
        raise ImportError("google.generativeai.GenerativeModel not found")
except ImportError as e:
    genai = None
    GenerationConfig = None
    GoogleAPIErrors = None
    logger.warning(
        f"google-generativeai package not found or incomplete: {e}. Gemini models will not be available."
    )

# Ollama
try:
    import ollama
except ImportError:
    ollama = None
    logger.warning("ollama package not found. Ollama models will not be available.")


# --- Constants ---
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.5
DEFAULT_API_TIMEOUT = 120  # Timeout in seconds for AI API calls

# Streamlit session state keys
SESSION_KEY_API_KEYS = "api_keys"


# --- API Key Encryption ---
# CRITICAL SECURITY NOTE: Using hardcoded Fernet key is insecure for distribution.
# Prioritize environment variables, but provide a documented fallback for local dev.
# Ensure the application fails gracefully if encryption cannot be initialized.
fernet = None
try:
    if Fernet is None:
        logger.error("Fernet encryption not available, API keys will not be encrypted")
    else:
        fernet_key_env = os.environ.get("OPENCLIP_FERNET_KEY")
        if fernet_key_env:
            try:
                fernet = Fernet(fernet_key_env.encode())
                logger.info("Using Fernet key from OPENCLIP_FERNET_KEY environment variable.")
            except ValueError as e:
                logger.error(
                    f"Invalid Fernet key format in environment variable OPENCLIP_FERNET_KEY: {e}"
                )
                raise RuntimeError(
                    "Invalid encryption key provided via environment variable."
                ) from e
        else:
            # WARNING: This hardcoded key is INSECURE for distributed applications.
            # It's provided as a fallback for local development ONLY.
            logger.warning(
                "Environment variable OPENCLIP_FERNET_KEY not set. "
                "Using hardcoded Fernet key. THIS IS INSECURE for production/distribution. "
                "Set the environment variable for better security."
            )
            # This is an example key, generate a unique one for your actual use.
            fernet = Fernet(b"jCqj3uJ9pG-Y7tWd3xSgFm8zN5xKc1LhA2vRb0_PzEw=")

except Exception as e:
    logger.error(
        f"Failed to initialize Fernet encryption system: {e}", exc_info=True
    )
    # Application behavior depends on whether encryption is critical.
    # Here, we disable saving/loading keys if encryption fails.
    if st:  # Check if streamlit is available
        st.error(
            "Encryption setup failed. API keys cannot be saved or loaded securely. "
            "Please check logs and ensure a valid encryption key is available."
        )


# Encryption helper functions
def encrypt_data(data: str) -> str:
    """Encrypts string data using the initialized Fernet instance."""
    if not fernet:
        logger.error("Fernet not initialized, cannot encrypt.")
        raise RuntimeError("Encryption service is unavailable.")
    return fernet.encrypt(data.encode()).decode()


def decrypt_data(data: str) -> str:
    """Decrypts string data using the initialized Fernet instance."""
    if not fernet:
        logger.error("Fernet not initialized, cannot decrypt.")
        raise RuntimeError("Encryption service is unavailable.")
    return fernet.decrypt(data.encode()).decode()


# --- API Key Management ---
class APIKeyManager:
    """
    Manages API keys, including loading from environment, file (encrypted),
    and saving securely to file. Integrates with Streamlit session state.
    """

    def __init__(self, key_file: str = API_KEY_FILE):
        self.key_file = key_file
        # Initialize session state key if it doesn't exist
        if SESSION_KEY_API_KEYS not in st.session_state:
            st.session_state[SESSION_KEY_API_KEYS] = {}
        self._load_keys()

    def _load_keys(self):
        """
        Initializes API keys in session state.
        Priority: Session State > Environment Variables > Encrypted File.
        Only loads from file if encryption is available.
        """
        # 1. Load from Environment Variables (non-persistent, overrides file on init)
        env_keys = self._get_keys_from_env()
        if env_keys:
            logger.info(f"Found {len(env_keys)} API keys in environment variables.")
            # Update session state, prioritizing env vars only if not already set in session
            for provider, key in env_keys.items():
                if provider not in st.session_state[SESSION_KEY_API_KEYS]:
                    st.session_state[SESSION_KEY_API_KEYS][provider] = key

        # 2. Load from Encrypted File (persistent, if encryption is enabled)
        if not fernet:
            logger.warning(
                "Encryption not available. Skipping loading/saving keys from file."
            )
            return  # Cannot load/save encrypted keys

        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        except OSError as e:
            logger.error(
                f"Failed to create directory for API key file {self.key_file}: {e}",
                exc_info=True,
            )
            # Don't halt execution, but keys won't be saved/loaded from file
            return

        if os.path.exists(self.key_file):
            loaded_file_keys = {}
            try:
                with open(self.key_file, "r") as f:
                    encrypted_keys = json.load(f)
                # Decrypt keys
                for k, v in encrypted_keys.items():
                    try:
                        loaded_file_keys[k] = decrypt_data(v)
                    except Exception as decrypt_e:
                        logger.error(
                            f"Failed to decrypt key for provider {k} from {self.key_file}: {decrypt_e}. Skipping this key."
                        )
                logger.info(
                    f"Loaded {len(loaded_file_keys)} API keys from {self.key_file}"
                )
                # Update session state, prioritizing existing session/env keys
                for provider, key in loaded_file_keys.items():
                    if provider not in st.session_state[SESSION_KEY_API_KEYS]:
                        st.session_state[SESSION_KEY_API_KEYS][provider] = key

            except (json.JSONDecodeError, OSError, RuntimeError, Exception) as e:
                logger.error(
                    f"Error loading or decrypting API keys from {self.key_file}: {e}",
                    exc_info=True,
                )
                st.warning(
                    f"Could not load API keys from file ({e}). File might be corrupt or encryption key changed."
                )
                # Do not reset session state here, keep env vars if loaded
        else:
            logger.info(
                f"API key file not found at {self.key_file}. Will create if keys are saved."
            )

    def _get_keys_from_env(self) -> Dict[str, str]:
        """Retrieves API keys from common environment variables."""
        env_keys = {}
        # Map internal provider names (lowercase) to potential environment variables
        # List multiple common names per provider.
        provider_env_map = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            # Add other providers and their typical env var names
        }

        for provider, env_vars in provider_env_map.items():
            for env_var in env_vars:
                key = os.environ.get(env_var)
                if key:
                    env_keys[provider] = key
                    break  # Use the first found key for this provider
        return env_keys

    def save_key(self, provider: str, api_key: str) -> bool:
        """
        Saves or updates an API key for a provider in session state and
        persists all session keys (except env vars) to the encrypted file.
        """
        if not provider or not api_key:
            logger.warning("Attempted to save empty provider or API key.")
            st.warning("Provider name and API key cannot be empty.")
            return False

        provider_lower = provider.lower().strip()
        # Update session state
        st.session_state.setdefault(SESSION_KEY_API_KEYS, {})[
            provider_lower
        ] = api_key
        logger.debug(f"API key for provider '{provider_lower}' updated in session state.")

        # Persist keys from session state to file (if encryption is available)
        return self._save_all_keys_to_file()

    def _save_all_keys_to_file(self) -> bool:
        """Helper to encrypt and save all keys from session state to the file."""
        if not fernet:
            st.error("Encryption is not available. Cannot save API keys securely.")
            return False

        try:
            # Get keys currently in session state
            session_keys = st.session_state.get(SESSION_KEY_API_KEYS, {})
            # Identify keys that came from environment variables (these should not be saved to file)
            env_keys = self._get_keys_from_env()
            keys_to_save = {}
            for provider, key in session_keys.items():
                # Save key if it's not empty AND it wasn't sourced from an env var
                # (or if it was modified from the env var value)
                if key and (provider not in env_keys or env_keys[provider] != key):
                    keys_to_save[provider] = encrypt_data(key)

            if not keys_to_save:
                # If no keys to save, delete the file if it exists
                if os.path.exists(self.key_file):
                    os.remove(self.key_file)
                    logger.info(
                        f"Removed API key file {self.key_file} as there are no keys to save."
                    )
                return True  # Nothing to save, operation successful

            # Ensure directory exists before writing
            os.makedirs(os.path.dirname(self.key_file), exist_ok=True)

            with open(self.key_file, "w") as f:
                json.dump(keys_to_save, f, indent=2)
            logger.info(f"Successfully saved {len(keys_to_save)} API keys to file.")
            return True

        except (OSError, TypeError, RuntimeError, ValueError) as e:
            logger.error(
                f"Error saving API keys to {self.key_file}: {e}", exc_info=True
            )
            st.error(f"Error saving API keys: {e}")
            return False

    def get_key(self, provider: str) -> Optional[str]:
        """
        Gets the API key for a provider from session state.
        Session state is the single source of truth after initialization.
        """
        provider_lower = provider.lower().strip()

        # Ollama and LLaVA typically run locally and don't require an API key.
        if provider_lower == "ollama" or provider_lower == "llava":
            return "local_access_no_key_needed"  # Use a non-None placeholder

        key = st.session_state.get(SESSION_KEY_API_KEYS, {}).get(provider_lower)

        if key:
            # logger.debug(f"Retrieved API key for '{provider_lower}' from session state.")
            return key
        else:
            # logger.debug(f"API key for provider '{provider_lower}' not found in session state.")
            return None

    def delete_key(self, provider: str) -> bool:
        """Deletes an API key for a provider from session state and the encrypted file."""
        provider_lower = provider.lower().strip()
        key_deleted = False

        # Delete from session state
        if provider_lower in st.session_state.get(SESSION_KEY_API_KEYS, {}):
            del st.session_state[SESSION_KEY_API_KEYS][provider_lower]
            key_deleted = True
            logger.debug(
                f"API key for provider '{provider_lower}' deleted from session state."
            )

        # Persist the changes to the file
        if self._save_all_keys_to_file():
            if key_deleted:
                logger.info(
                    f"Successfully removed API key for '{provider_lower}' from persistent storage."
                )
            return True
        else:
            # If saving failed, the key might still be deleted from session but not file
            logger.error(
                f"Failed to update persistent storage after deleting key for '{provider_lower}'."
            )
            return False

    def list_providers_with_keys(self) -> List[str]:
        """Returns a sorted list of provider names (lowercase) for which keys are available."""
        return sorted(list(st.session_state.get(SESSION_KEY_API_KEYS, {}).keys()))

    def is_key_from_env(self, provider: str) -> bool:
        """
        Returns True if the API key for the given provider is set via an environment variable.
        For Ollama/LLaVA, always returns False.
        """
        provider_lower = provider.lower().strip()
        # Ollama/LLaVA do not use API keys
        if provider_lower == "ollama" or provider_lower == "llava":
            return False

        # Map provider to possible environment variable names
        provider_env_map = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            # Add more providers as needed
        }
        env_vars = provider_env_map.get(provider_lower, [])
        for env_var in env_vars:
            if os.environ.get(env_var):
                return True
        return False


# --- Provider-Specific Integrations ---
# Wrappers can encapsulate complex initialization or API call logic.
class GeminiIntegration:
    """
    Handles interaction with the Google Generative AI (Gemini) client.
    Note: The underlying `genai.configure` is global. This class manages
    setting the key for the application's context.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.genai = genai  # Use the globally imported module
        self.initialized = False
        self._cached_models = None
        self._cache_timestamp = 0
        self._cache_duration = 300  # Cache for 5 minutes
        
        if self.genai and self.api_key:
            self._configure()
        elif not self.genai:
            logger.error(
                "google.generativeai module not imported. Cannot initialize Gemini."
            )
        # If api_key is None, it will be configured later if needed

    def _configure(self):
        """Configures the global `genai` client with the stored API key."""
        if not self.genai or not self.api_key:
            logger.warning("Cannot configure Gemini: genai module or API key missing.")
            return False

        try:
            # This configures the genai module globally for subsequent calls.
            genai.configure(api_key=self.api_key)
            self.initialized = True
            logger.info("Google Generative AI (Gemini) API configured successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to configure Google Gemini API: {e}", exc_info=True)
            self.initialized = False
            return False

    def set_api_key(self, api_key: str) -> bool:
        """Sets or updates the API key and attempts to reconfigure."""
        self.api_key = api_key
        return self._configure()

    async def analyze_with_gemini(
        self,
        model_name: str,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_API_TIMEOUT,
    ) -> str:
        """
        Analyzes content using a Gemini model.

        Args:
            model_name: The specific Gemini model (e.g., "gemini-1.5-pro-latest").
            prompt: The text prompt.
            images: List of base64 encoded image strings.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            timeout: Request timeout in seconds.

        Returns:
            The model's response text or an error string prefixed with "Error:".
        """
        if not self.initialized or not self.genai:
            # Attempt to re-configure if not initialized but key exists
            if not self.initialized and self.api_key and self._configure():
                logger.info("Re-configured Gemini client before analysis.")
            else:
                logger.error(
                    "Gemini API is not configured or key is missing. Cannot perform analysis."
                )
                return "Error: Gemini API not configured or API key missing."

        try:
            model = genai.GenerativeModel(model_name)
            content_parts = []

            # Prepare image parts
            if images:
                for img_base64 in images:
                    try:
                        # Gemini expects raw bytes for images
                        img_bytes = base64.b64decode(img_base64)
                        # Assuming JPEG format based on previous code context
                        content_parts.append(
                            {"mime_type": "image/jpeg", "data": img_bytes}
                        )
                    except (binascii.Error, ValueError) as img_err:
                        logger.error(
                            f"Error decoding base64 image for Gemini: {img_err}",
                            exc_info=True,
                        )
                        return f"Error: Invalid base64 image data provided for Gemini."

            # Add text part
            content_parts.append({"text": prompt})

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                # Add other relevant config if needed (top_p, top_k)
            )

            # Default safety settings are usually sufficient, but can be customized:
            # safety_settings = [...]

            # The genai library's async methods handle timeouts internally based on config/defaults,
            # but we wrap with asyncio.wait_for for an additional layer of control.
            response = await asyncio.wait_for(
                model.generate_content_async(
                    content_parts,
                    generation_config=generation_config,
                    # safety_settings=safety_settings, # Uncomment if customizing
                    request_options={"timeout": timeout},  # Pass timeout to underlying request
                ),
                timeout=timeout + 5,  # Outer timeout slightly larger
            )

            # Process response safely
            if not response.candidates:
                block_reason = "Unknown"
                try:
                    # Access prompt feedback safely
                    if response.prompt_feedback:
                        if hasattr(response.prompt_feedback, "block_reason"):
                            block_reason = response.prompt_feedback.block_reason.name
                        elif hasattr(response.prompt_feedback, "safety_ratings"):
                            blocked_categories = [
                                r.category.name
                                for r in response.prompt_feedback.safety_ratings
                                if "BLOCK" in r.probability.name
                            ]
                            if blocked_categories:
                                block_reason = (
                                    f"Safety Block ({', '.join(blocked_categories)})"
                                )
                except Exception as feedback_err:
                    logger.warning(
                        f"Could not determine block reason from prompt_feedback: {feedback_err}"
                    )

                logger.warning(
                    f"Gemini response blocked or empty for model {model_name}. Reason: {block_reason}"
                )
                return f"Error: Gemini response blocked (Reason: {block_reason})"

            # Extract text safely from the first candidate
            response_text = ""
            try:
                first_candidate = response.candidates[0]
                if first_candidate.content and first_candidate.content.parts:
                    response_text = "".join(
                        part.text
                        for part in first_candidate.content.parts
                        if hasattr(part, "text")
                    )
            except (IndexError, AttributeError) as extract_err:
                logger.error(
                    f"Error extracting text from Gemini response structure: {extract_err}",
                    exc_info=True,
                )
                return "Error: Could not parse Gemini response content."

            if not response_text:
                logger.warning(
                    f"Gemini response content is empty for model {model_name}."
                )
                # Check finish reason if response is empty
                finish_reason = "Unknown"
                try:
                    finish_reason = response.candidates[0].finish_reason.name
                except (IndexError, AttributeError):
                    pass
                return f"Error: Received empty content from Gemini (Finish Reason: {finish_reason})."

            return response_text

        except GoogleAPIErrors.DeadlineExceeded as e:
            logger.error(
                f"Gemini API call timed out ({timeout}s) for model {model_name}: {e}"
            )
            return f"Error: Gemini API call timed out ({timeout}s)."
        except GoogleAPIErrors.GoogleAPIError as e:
            logger.error(f"Gemini API error for model {model_name}: {e}", exc_info=True)
            # Provide more specific error info if available (e.g., permission denied)
            return f"Error: Gemini API Error - {e}"
        except asyncio.TimeoutError:
            logger.error(
                f"Gemini async call timed out after {timeout+5}s (wrapper timeout) for model {model_name}"
            )
            return f"Error: Gemini operation timed out ({timeout+5}s)."
        except Exception as e:
            logger.error(
                f"Unexpected error analyzing with Gemini model {model_name}: {e}",
                exc_info=True,
            )
            return f"Error: Unexpected error during Gemini analysis ({type(e).__name__})."

    def list_available_models(self) -> List[str]:
        """
        Fetch available Gemini models from the API with caching.
        
        Returns:
            List of available model names, or empty list if failed.
        """
        # Check cache validity
        current_time = time.time()
        if (self._cached_models is not None and 
            current_time - self._cache_timestamp < self._cache_duration):
            logger.debug("Using cached Gemini models list")
            return self._cached_models
        
        # Fetch fresh models from API
        if not self.initialized or not self.genai:
            logger.warning("Gemini not configured, cannot fetch models")
            return []
        
        try:            # Use genai.list_models() to fetch available models
            models = genai.list_models()
            model_names = []
            
            for model in models:
                # Extract model name from the full model path
                # Model names come as "models/gemini-1.5-pro" format
                if hasattr(model, 'name'):
                    model_name = model.name
                    if model_name.startswith('models/'):
                        model_name = model_name[7:]  # Remove "models/" prefix
                    model_names.append(model_name)
            
            # Cache the results
            self._cached_models = model_names
            self._cache_timestamp = current_time
            
            logger.info(f"Successfully fetched {len(model_names)} Gemini models from API")
            return model_names
            
        except Exception as e:
            logger.error(f"Failed to fetch Gemini models from API: {e}", exc_info=True)
            return []

    def get_model_info_from_api(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific model from the API.
        
        Args:
            model_name: The model name to get info for
            
        Returns:
            Dictionary with model info or None if failed
        """
        if not self.initialized or not self.genai:
            return None
            
        try:
            # Get model info from API
            full_model_name = f"models/{model_name}" if not model_name.startswith("models/") else model_name
            model_info = genai.get_model(full_model_name)
            
            # Extract relevant information and convert to our format
            capabilities = ["text"]  # All Gemini models support text
            
            # Check if model supports vision (images)
            if hasattr(model_info, 'supported_generation_methods'):
                if any('Vision' in str(method) for method in model_info.supported_generation_methods):
                    capabilities.append("image")
            else:
                # For newer models, assume vision capability if name suggests it
                if any(keyword in model_name.lower() for keyword in ['vision', 'pro', 'flash']):
                    capabilities.append("image")
            
            # Determine quality tier based on model name
            quality = "standard"
            if "pro" in model_name.lower():
                quality = "premium"
            elif "flash" in model_name.lower():
                quality = "fast"
            
            return {
                "provider": "Google",
                "type": "vision" if "image" in capabilities else "text",
                "quality": quality,
                "requires_api_key": True,
                "capabilities": capabilities,
                "dynamic": True,  # Mark as dynamically fetched
            }
            
        except Exception as e:
            logger.debug(f"Could not fetch detailed info for model {model_name}: {e}")
            return None


# --- AI Model Registry ---
class ModelRegistry:
    """Provides information about known AI models and their capabilities."""

    # Static definition of known models. Keyed by lowercase provider name.
    # Includes metadata like type, quality, key requirement, and capabilities.
    KNOWN_MODELS = {
        "openai": {
            "gpt-4o": {
                "provider": "OpenAI",
                "type": "vision",
                "quality": "premium",
                "requires_api_key": True,
                "capabilities": ["text", "image", "json"],
            },
            "gpt-4o-mini": {
                "provider": "OpenAI",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text", "image", "json"],
            },
            "gpt-4-turbo": {
                "provider": "OpenAI",
                "type": "vision",
                "quality": "premium",
                "requires_api_key": True,
                "capabilities": ["text", "image", "json"],
            },
            "gpt-3.5-turbo": {
                "provider": "OpenAI",
                "type": "text",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text", "json"], # Check if 3.5 supports JSON mode
            },
        },
        "google": {
            "gemini-1.5-pro-latest": {
                "provider": "Google",
                "type": "vision",
                "quality": "premium",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
            "gemini-1.5-flash-latest": {
                "provider": "Google",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
            "gemini-pro-vision": { # Older model name, might be deprecated
                "provider": "Google",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
            "gemini-pro": {
                "provider": "Google",
                "type": "text",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text"],
            },
        },
        "anthropic": {
            "claude-3-opus-20240229": {
                "provider": "Anthropic",
                "type": "vision",
                "quality": "premium",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
            "claude-3-sonnet-20240229": {
                "provider": "Anthropic",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
            "claude-3-haiku-20240307": {
                "provider": "Anthropic",
                "type": "vision",
                "quality": "fast",
                "requires_api_key": True,
                "capabilities": ["text", "image"],
            },
             "claude-2.1": { # Older model
                "provider": "Anthropic",
                "type": "text",
                "quality": "standard",
                "requires_api_key": True,
                "capabilities": ["text"],
            },
        },
        "ollama": {
            # Ollama models are often dynamically discovered.
            # Define common ones here for UI hints and capability checks.
            # 'dynamic': True indicates these might need confirmation via Ollama API.
            "llama3": {
                "provider": "Ollama",
                "type": "vision", # Llama3 base is text, vision variant likely exists
                "quality": "standard",
                "requires_api_key": False,
                "capabilities": ["text", "image"], # Check specific Llama3 variant
                "dynamic": True,
            },
            "llava": {
                "provider": "Ollama",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": False,
                "capabilities": ["text", "image"],
                "dynamic": True,
                "is_free": True,  # Explicitly mark as free
            },
            "llava:latest": {
                "provider": "Ollama",
                "type": "vision",
                "quality": "standard",
                "requires_api_key": False,
                "capabilities": ["text", "image"],
                "dynamic": True,
                "is_free": True,  # Explicitly mark as free
            },
            "mistral": {
                "provider": "Ollama",
                "type": "text",
                "quality": "standard",
                "requires_api_key": False,
                "capabilities": ["text"],
                "dynamic": True,
            },
        },
        # Add other providers (e.g., HuggingFace) here if needed
    }

    def __init__(self, key_manager: Optional['APIKeyManager'] = None):
        """Initialize ModelRegistry with optional dynamic model support."""
        self.key_manager = key_manager
        self._gemini_integration = None
        self._dynamic_models_cache = {}
        self._cache_timestamps = {}
        self._cache_duration = 300  # 5 minutes cache

    def _get_gemini_integration(self):
        """Get or create a GeminiIntegration instance for dynamic model fetching."""
        if not self._gemini_integration and self.key_manager:
            api_key = self.key_manager.get_key("google")
            if api_key:
                self._gemini_integration = GeminiIntegration(api_key)
        return self._gemini_integration

    def _get_dynamic_gemini_models(self) -> List[Dict]:
        """Fetch dynamic Gemini models from the API."""
        # Check cache first
        current_time = time.time()
        if ("google" in self._dynamic_models_cache and 
            current_time - self._cache_timestamps.get("google", 0) < self._cache_duration):
            return self._dynamic_models_cache["google"]

        dynamic_models = []
        gemini_integration = self._get_gemini_integration()
        
        if gemini_integration and gemini_integration.initialized:
            try:
                model_names = gemini_integration.list_available_models()
                logger.info(f"Fetched {len(model_names)} dynamic Gemini models from API")
                
                for model_name in model_names:
                    # Get detailed info for each model
                    model_info = gemini_integration.get_model_info_from_api(model_name)
                    if model_info:
                        dynamic_models.append({
                            "model": model_name,
                            "provider": "Google",
                            **model_info
                        })
                    else:
                        # Fallback info if detailed fetch fails
                        dynamic_models.append({
                            "model": model_name,
                            "provider": "Google",
                            "type": "vision",
                            "quality": "standard",
                            "requires_api_key": True,
                            "capabilities": ["text", "image"],
                            "dynamic": True
                        })
                
                # Cache the results
                self._dynamic_models_cache["google"] = dynamic_models
                self._cache_timestamps["google"] = current_time
                
            except Exception as e:
                logger.warning(f"Failed to fetch dynamic Gemini models: {e}")
        
        return dynamic_models

    def list_models_for_provider(self, provider_name_or_key: str) -> List[str]:
        """Lists model names for a specific provider, including dynamic models."""
        provider_key = provider_name_or_key.lower()
        
        # Get static models
        static_models = sorted(list(self.KNOWN_MODELS.get(provider_key, {}).keys()))
        
        # For Google/Gemini, also fetch dynamic models
        if provider_key == "google":
            dynamic_models = self._get_dynamic_gemini_models()
            dynamic_model_names = [m["model"] for m in dynamic_models]
            
            # Combine static and dynamic models, remove duplicates
            all_models = static_models + [name for name in dynamic_model_names if name not in static_models]
            return sorted(all_models)
        
        return static_models

    def get_model_info(self, provider_name_or_key: str, model_name: str) -> Optional[Dict]:
        """Gets detailed info for a specific model, checking both static and dynamic sources."""
        provider_key = provider_name_or_key.lower()
        
        # First check static models
        static_info = self.KNOWN_MODELS.get(provider_key, {}).get(model_name)
        if static_info:
            return static_info
        
        # For Google/Gemini, check dynamic models
        if provider_key == "google":
            dynamic_models = self._get_dynamic_gemini_models()
            for model in dynamic_models:
                if model.get("model") == model_name:
                    # Return a copy without the "model" key to match expected format
                    model_info = {k: v for k, v in model.items() if k != "model"}
                    return model_info
        
        # Standard lookup for models without tags
        return self.KNOWN_MODELS.get(provider_key, {}).get(model_name)


    def list_vision_models(self) -> List[Dict]:
        """Lists all models supporting image input, including dynamic models."""
        vision_models = []
        
        # Get static vision models
        for provider_key, models in self.KNOWN_MODELS.items():
            for name, info in models.items():
                if "image" in info.get("capabilities", []):
                    vision_models.append(
                        {"provider": info.get("provider", provider_key.capitalize()), "model": name, **info}
                    )
        
        # Add dynamic Gemini vision models
        dynamic_gemini_models = self._get_dynamic_gemini_models()
        for model in dynamic_gemini_models:
            if "image" in model.get("capabilities", []):
                vision_models.append(model)
        
        return vision_models

    def list_all_models_structured(self) -> Dict[str, List[Dict]]:
        """Returns a structured dict including dynamic models."""
        structured = {}
        
        # Process static models
        for provider_key, models in self.KNOWN_MODELS.items():
            provider_display_name = provider_key.capitalize()
            if models:
                provider_display_name = next(iter(models.values())).get("provider", provider_display_name)

            model_list = []
            for name, info in models.items():
                model_list.append({"model": name, **info})

            if model_list:
                structured[provider_display_name] = sorted(model_list, key=lambda m: m['model'])
        
        # Add dynamic Gemini models
        dynamic_gemini_models = self._get_dynamic_gemini_models()
        if dynamic_gemini_models:
            google_models = structured.get("Google", [])
            # Add dynamic models that aren't already in static list
            existing_model_names = {m["model"] for m in google_models}
            for model in dynamic_gemini_models:
                if model["model"] not in existing_model_names:
                    google_models.append(model)
            
            structured["Google"] = sorted(google_models, key=lambda m: m['model'])
        
        return structured

    def list_free_models(self) -> List[Dict]:
        """Returns a list of models that are free to use (no API key required or explicitly marked as free)."""
        free_models = []
        
        # Check static models
        for provider_key, models in self.KNOWN_MODELS.items():
            for name, info in models.items():
                # Consider a model free if it's explicitly marked as free OR doesn't require an API key
                is_free = info.get("is_free", False) or not info.get("requires_api_key", True)
                if is_free:
                    free_models.append({
                        "provider": info.get("provider", provider_key.capitalize()),
                        "model": name,
                        **info
                    })
        
        # Note: Dynamic Gemini models typically require API keys, so they won't be included
        # unless specifically marked as free in the API response
        
        return free_models


# --- AI Client Factory ---
class AIClientFactory:
    """
    Factory to create and cache AI client instances based on provider.
    Handles API key retrieval and basic client configuration.
    """

    def __init__(self, key_manager: APIKeyManager, model_registry: ModelRegistry):
        self.key_manager = key_manager
        self.model_registry = model_registry
        # Cache for initialized clients { 'provider_lower': client_instance }
        self._client_cache: Dict[str, Any] = {}
        # Cache for dynamically checked Ollama models
        self._ollama_local_models: Optional[List[str]] = None

    async def get_client(self, provider: str, model_name: Optional[str] = None) -> Any:
        """
        Gets or creates an AI client for the specified provider.
        Handles API key checks and basic client initialization.

        Args:
            provider: The provider name (e.g., "OpenAI", "google").
            model_name: Optional model name to check capabilities/requirements.

        Returns:
            An initialized client instance (e.g., AsyncOpenAI, GeminiIntegration).

        Raises:
            ValueError: If provider is unsupported or API key is required but not found.
            ImportError: If the required SDK package is not installed.
            ConnectionError: If connection to a local service (like Ollama) fails.
            RuntimeError: For unexpected initialization errors.
        """
        provider_lower = provider.lower().strip()

        # Return cached client if available and valid
        if provider_lower in self._client_cache:
            # Add checks here if client validity can expire (e.g., token refresh)
            # Or if API key changed since initialization (requires more complex state tracking)
            return self._client_cache[provider_lower]

        # Determine API key requirement
        requires_api_key = True  # Default assumption
        if provider_lower == "ollama":
            requires_api_key = False
        elif model_name:
            model_info = self.model_registry.get_model_info(provider_lower, model_name)
            if model_info:
                requires_api_key = model_info.get("requires_api_key", True)
        # If no model_name provided, assume key is needed unless it's Ollama

        api_key = None
        if requires_api_key:
            api_key = self.key_manager.get_key(provider_lower)
            if not api_key:
                raise ValueError(f"API key required for provider '{provider}' but not found.")

        # Initialize client based on provider
        client = None
        try:
            if provider_lower == "openai":
                if not AsyncOpenAI:
                    raise ImportError("OpenAI SDK not installed.")
                client = AsyncOpenAI(api_key=api_key, timeout=DEFAULT_API_TIMEOUT)

            elif provider_lower == "anthropic":
                if not anthropic:
                    raise ImportError("Anthropic SDK not installed.")
                client = anthropic.AsyncAnthropic(api_key=api_key, timeout=DEFAULT_API_TIMEOUT)

            elif provider_lower == "google":
                if not genai:
                    raise ImportError("Google Generative AI SDK not installed.")
                # The GeminiIntegration wrapper handles the global genai.configure
                client = GeminiIntegration(api_key=api_key)
                if not client.initialized:
                    # Error logged within GeminiIntegration, raise specific error here
                    raise ConnectionError("Failed to configure Google Gemini client. Check API key and permissions.")

            elif provider_lower == "ollama":
                if not ollama:
                    raise ImportError("Ollama SDK not installed.")
                # Prefer AsyncClient if available
                try:
                    if hasattr(ollama, "AsyncClient"):
                        client = ollama.AsyncClient(timeout=DEFAULT_API_TIMEOUT)
                        # Perform a quick check to ensure the server is reachable
                        await client.list()
                        logger.info("Ollama AsyncClient connected successfully.")
                    else:
                        # Fallback sync client (requires running calls in executor)
                        client = ollama.Client(timeout=DEFAULT_API_TIMEOUT)
                        # Test connection (this blocks)
                        client.list()
                        logger.warning(
                            "Using synchronous Ollama client. API calls will run in thread executor."
                        )
                        logger.info("Ollama Sync Client connected successfully.")
                except (ollama.RequestError, Exception) as ollama_err:
                    logger.error(f"Ollama connection failed: {ollama_err}", exc_info=True)
                    raise ConnectionError(
                        f"Could not connect to Ollama server. Ensure it's running. Error: {ollama_err}"
                    )

            # Add other providers like HuggingFace using aiohttp or requests here
            # elif provider_lower == "huggingface":
            #    # Requires managing aiohttp session, potentially storing it in self._client_cache
            #    pass

            else:
                raise ValueError(f"Unsupported provider: '{provider}'")

            # Cache the successfully created client
            self._client_cache[provider_lower] = client
            logger.info(f"Initialized AI client for provider: '{provider_lower}'")
            return client

        except (ValueError, ImportError, ConnectionError) as e:
            # Re-raise known, expected errors
            raise e
        except Exception as e:
            # Catch unexpected errors during initialization
            logger.error(
                f"Unexpected error initializing client for '{provider_lower}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Failed to initialize client for {provider}: {e}") from e

    async def check_ollama_model_availability(self, model_name: str) -> bool:
        """Check if an Ollama model is available locally."""
        debug_print(f"Checking if Ollama model '{model_name}' is available locally")
        
        try:
            # First try to get list of models via API
            client = await self.get_client("ollama")
            if hasattr(client, "list") and callable(client.list):
                models_list = await client.list()
                
                if isinstance(models_list, dict) and "models" in models_list:
                    models = models_list.get("models", [])
                    model_names = [m.get("name") for m in models if isinstance(m, dict) and "name" in m]
                elif isinstance(models_list, list):
                    model_names = [m.get("name") for m in models_list if isinstance(m, dict) and "name" in m]
                else:
                    logger.error(f"Failed to list local Ollama models: {type(models_list)}")
                    model_names = []
                
                logger.info(f"Checking for model '{model_name}' in fetched list: {model_names}")
                
                # Handle special case for "llava", which might exist with various suffixes
                if model_name.startswith("llava"):
                    debug_print(f"Special handling for {model_name} - checking multiple variations")
                    # Check direct match first
                    if model_name in model_names:
                        return True
                    
                    # Try alternatives like llava:latest or just llava
                    if "llava:latest" in model_names or "llava" in model_names:
                        debug_print(f"Found alternative LLaVA model in model list")
                        return True
                    
                    # Alternative approach: run direct command to check
                    try:
                        import subprocess
                        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
                        output = result.stdout.lower()
                        # Check if output contains llava
                        contains_llava = "llava" in output
                        debug_print(f"Direct command output contains 'llava': {contains_llava}")
                        if contains_llava:
                            debug_print(f"Found LLaVA model via direct command")
                            return True
                    except Exception as e:
                        logger.warning(f"Error checking Ollama models directly: {e}")
                        # Continue with regular checking
                
                debug_print(f"Ollama model '{model_name}' available: {model_name in model_names}")
                return model_name in model_names
            else:
                logger.warning(f"Ollama client doesn't have 'list' method. Using fallback approach.")
                # Fallback: check if ollama is running
                try:
                    import requests
                    # Try to connect to Ollama API
                    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
                    response = requests.get(f"{ollama_host}/api/version", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Ollama server is running, assuming model may be available")
                        # We can't confirm the model exists, but we know Ollama is running
                        return True
                    else:
                        logger.warning(f"Ollama server returned status {response.status_code}")
                        return False
                except Exception as conn_err:
                    logger.warning(f"Failed to connect to Ollama server: {conn_err}")
                    return False
        except Exception as e:
            logger.warning(f"Error checking Ollama model availability: {e}")
            # Handle error gracefully - assume model doesn't exist rather than breaking
            return False

    async def close_all(self):
        """Closes any open client sessions or connections."""
        logger.info("Closing AI client resources...")
        for provider, client in self._client_cache.items():
            try:
                # Add specific close logic for clients that need it (e.g., aiohttp sessions)
                # if isinstance(client, aiohttp.ClientSession) and not client.closed:
                #     await client.close()
                #     logger.info(f"Closed aiohttp client session for {provider}")

                # For async clients like OpenAI/Anthropic/Ollama, explicit close might not be needed
                # unless they manage persistent connections not handled by the HTTP library.
                # Check documentation if unsure. Example:
                # if hasattr(client, 'close') and asyncio.iscoroutinefunction(client.close):
                #    await client.close()
                pass # No explicit close needed for current clients
            except Exception as e:
                logger.error(f"Error closing client for provider {provider}: {e}", exc_info=True)
        self._client_cache = {}
        self._ollama_local_models = None # Clear cache on close
        logger.info("AI client resources closed.")


# --- AI Model Manager ---
class AIModelManager:
    """
    Provides a unified asynchronous interface for interacting with different AI models.
    Handles client retrieval, request formatting, API calls, and basic response parsing.
    """

    def __init__(self, key_manager: APIKeyManager, model_registry: ModelRegistry):
        self.key_manager = key_manager
        self.model_registry = model_registry
        self.client_factory = AIClientFactory(key_manager, model_registry)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4, thread_name_prefix="SyncAIExecutor"
        ) # For running sync Ollama calls

    async def analyze_with_model(
        self,
        provider: str,
        model_name: str,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = DEFAULT_API_TIMEOUT,
        format_type: str = "text",  # 'text' or 'json'
    ) -> str:
        """
        Runs analysis using a specified AI model.

        Args:
            provider: AI provider name (case-insensitive).
            model_name: Specific model name.
            prompt: Text prompt for the model.
            images: Optional list of base64 encoded image strings for vision models.
            temperature: Controls randomness (0.0 to ~2.0).
            max_tokens: Max tokens for the response.
            timeout: Request timeout in seconds.
            format_type: Desired output format ('text' or 'json'). JSON mode is
                         only attempted if the model registry indicates support.

        Returns:
            The AI's response as a string, or an error message prefixed with "Error:".
        """
        print(f"DEBUG: analyze_with_model called - provider={provider}, model={model_name}, has_images={images is not None}")
        
        provider_lower = provider.lower().strip()
        model_info = self.model_registry.get_model_info(provider_lower, model_name)

        if not model_info:
            print(f"DEBUG: Model info not found for {provider_lower}:{model_name}")
            return f"Error: Model '{model_name}' not registered for provider '{provider}'."

        # Validate image input against model capabilities
        model_capabilities = model_info.get("capabilities", [])
        print(f"DEBUG: Model capabilities: {model_capabilities}")
        if images and "image" not in model_capabilities:
            logger.warning(
                f"Model {provider}:{model_name} does not support image input, but images were provided. Ignoring images."
            )
            print(f"DEBUG: Ignoring images as model does not support vision")
            images = None  # Discard images for non-vision models

        # Handle JSON format request
        use_json_mode = (
            format_type == "json" and "json" in model_capabilities
        )
        if format_type == "json" and not use_json_mode:
            logger.warning(
                f"Model {provider}:{model_name} does not explicitly support JSON output mode via API. "
                "Will attempt to instruct via prompt, but formatting is not guaranteed."
            )
            # Modify prompt to request JSON (handled within specific provider methods if needed)
            prompt = self._ensure_json_instruction_in_prompt(prompt)

        # Get the client instance
        try:
            # For Ollama, also check if the specific model is available locally
            if provider_lower == "ollama":
                print(f"DEBUG: Checking if Ollama model '{model_name}' is available locally")
                is_available = await self.client_factory.check_ollama_model_availability(model_name)
                print(f"DEBUG: Ollama model '{model_name}' available: {is_available}")
                logger.info(f"Result of check_ollama_model_availability for '{model_name}': {is_available}") 
                if not is_available:
                    # For LLaVA specifically, provide more helpful error message
                    if "llava" in model_name.lower():
                        return f"""Error: LLaVA model '{model_name}' not found locally. 

To install it, run this command in your terminal:
```
ollama pull llava
```

If you're using Windows PowerShell:
```
& ollama pull llava
```

Make sure the Ollama service is running before attempting analysis."""
                    return f"""Error: Ollama model '{model_name}' not found locally. 

To install it, run this command in your terminal:
```
ollama pull {model_name}
```

If you're using Windows PowerShell:
```
& ollama pull {model_name}
```

Make sure the Ollama service is running before attempting analysis."""

            client = await self.client_factory.get_client(provider_lower, model_name)
            print(f"DEBUG: Got client for {provider_lower}:{model_name} - type: {type(client)}")

        except (ValueError, ConnectionError, ImportError, RuntimeError) as e:
            # Catch expected errors from client factory
            logger.error(f"Failed to get client for {provider}/{model_name}: {e}")
            print(f"DEBUG: Error getting client: {type(e).__name__}: {e}")
            return f"Error: Client initialization failed - {e}"
        except Exception as e:
            logger.error(
                f"Unexpected error getting client for {provider}/{model_name}: {e}",
                exc_info=True,
            )
            print(f"DEBUG: Unexpected error getting client: {type(e).__name__}: {e}")
            return f"Error: Failed to get AI client ({type(e).__name__})."

        # --- Call Provider-Specific Analysis Method ---
        try:
            if provider_lower == "openai":
                if not isinstance(client, AsyncOpenAI):
                     return f"Error: Invalid client type for OpenAI ({type(client)})."
                return await self._analyze_with_openai(
                    client, model_name, prompt, images, temperature, max_tokens, timeout, use_json_mode
                )

            elif provider_lower == "anthropic":
                 if not (anthropic and isinstance(client, anthropic.AsyncAnthropic)):
                     return f"Error: Invalid client type for Anthropic ({type(client)})."
                 return await self._analyze_with_anthropic(
                    client, model_name, prompt, images, temperature, max_tokens, timeout
                )

            elif provider_lower == "google":
                if not isinstance(client, GeminiIntegration):
                     return f"Error: Invalid client type for Google Gemini ({type(client)})."
                # GeminiIntegration's method handles its own async logic
                return await client.analyze_with_gemini(
                    model_name, prompt, images, temperature, max_tokens, timeout
                )

            elif provider_lower == "ollama":
                 if not (ollama and isinstance(client, (ollama.AsyncClient, ollama.Client))):
                      return f"Error: Invalid client type for Ollama ({type(client)})."
                 print(f"DEBUG: Calling _analyze_with_ollama for {provider_lower}:{model_name}")
                 return await self._analyze_with_ollama(
                    client, model_name, prompt, images, temperature, timeout, max_tokens
                )

            # Add other providers here...

            else:
                # Should have been caught by get_client, but acts as a safeguard
                print(f"DEBUG: No analysis logic implemented for provider '{provider}'")
                return f"Error: Analysis logic not implemented for provider '{provider}'."

        # Catch specific API errors from different SDKs
        except Exception as e:
            # Log the exception for debugging
            logger.error(f"Exception in analyze_with_model for {provider}/{model_name}: {type(e).__name__} - {e}")
            print(f"DEBUG: Exception in analyze_with_model: {type(e).__name__} - {e}")
            
            # Check for Pydantic serialization error common with image handling
            if "PydanticSerializationError" in str(type(e).__name__):
                return f"Error: Image format serialization failed for {provider}/{model_name}. This is often caused by an incompatible image format or size."
            
            # Handle different types of exceptions with more specific error messages
            if hasattr(e, "__module__") and "openai" in getattr(e, "__module__", ""):
                return f"Error: OpenAI API call failed ({type(e).__name__}). Details: {e}"
            elif hasattr(e, "__module__") and "anthropic" in getattr(e, "__module__", ""):
                return f"Error: Anthropic API call failed ({type(e).__name__}). Details: {e}"
            elif hasattr(e, "__module__") and "google" in getattr(e, "__module__", ""):
                return f"Error: Google API call failed ({type(e).__name__}). Details: {e}"
            elif hasattr(e, "__module__") and "ollama" in getattr(e, "__module__", ""):
                # Special handling for Ollama errors that might be related to LLaVA
                if "llava" in model_name.lower():
                    return f"Error: LLaVA model not found. Use 'ollama pull llava' to install it."
                return f"Error: Ollama API call failed ({type(e).__name__}). Details: {e}"
            elif isinstance(e, asyncio.TimeoutError):
                return f"Error: Operation timed out ({timeout}s)."
            elif isinstance(e, ConnectionError):
                return f"Error: Connection failed - {e}"
            elif "base exception" in str(type(e).__name__).lower() or str(e) == "catching classes that do not inherit from BaseException is not allowed":
                return f"Error: Invalid exception handling in underlying code. This might be a bug in the app."
            else:
                # Generic catch-all for unexpected exceptions
                return f"Error: An unexpected error occurred during analysis ({type(e).__name__}). Details: {e}"

    def _ensure_json_instruction_in_prompt(self, prompt: str) -> str:
        """Appends a JSON formatting instruction if not already present."""
        json_instruction = "\n\nRespond ONLY with a valid JSON object."
        # Simple check if the instruction or similar phrasing is already near the end
        if "json" not in prompt.lower()[-50:] and "json object" not in prompt.lower()[-50:]:
            logger.debug("Adding JSON formatting instruction to prompt.")
            return prompt + json_instruction
        return prompt

    # --- Provider-Specific Implementation Methods ---

    async def _analyze_with_openai(
        self,
        client: AsyncOpenAI,
        model_name: str,
        prompt: str,
        images: Optional[List[str]],
        temperature: float,
        max_tokens: int,
        timeout: int,
        use_json_mode: bool,
    ) -> str:
        """Calls the OpenAI Chat Completions API."""
        messages = []
        content_parts = [{"type": "text", "text": prompt}]
        if images:
            for img_b64 in images:
                # OpenAI Vision API expects data URLs (or external URLs)
                content_parts.append(
                    {
                        "type": "image_url",
                        # Assume JPEG, adjust if mime type is known otherwise
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    }
                )
        messages.append({"role": "user", "content": content_parts})

        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }

        if use_json_mode:
            request_params["response_format"] = {"type": "json_object"}
            logger.debug(f"Using OpenAI native JSON mode for {model_name}.")

        response = await client.chat.completions.create(**request_params)

        # Safely extract response content
        try:
            result = response.choices[0].message.content
            if result is None:
                 # Check finish reason
                 finish_reason = response.choices[0].finish_reason
                 logger.warning(f"OpenAI response content is None for model {model_name}. Finish Reason: {finish_reason}")
                 return f"Error: Received empty response from OpenAI (Finish Reason: {finish_reason})."
            return result.strip()
        except (IndexError, AttributeError) as e:
            logger.error(f"Error parsing OpenAI response: {e}. Response: {response}")
            return f"Error: Failed to parse OpenAI response structure."

    async def _analyze_with_anthropic(
        self,
        client: anthropic.AsyncAnthropic,
        model_name: str,
        prompt: str,
        images: Optional[List[str]],
        temperature: float,
        max_tokens: int,
        timeout: int, # Note: Anthropic timeout is set on client, not per-call easily
    ) -> str:
        """Calls the Anthropic Messages API."""
        content_parts = []
        if images:
            for img_b64 in images:
                content_parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg", # Assume JPEG
                            "data": img_b64,
                        },
                    }
                )
        # Text part must come last for Anthropic according to some docs/versions
        content_parts.append({"type": "text", "text": prompt})

        # Wrap the call with asyncio.wait_for to enforce timeout if client doesn't support per-call timeout
        response = await asyncio.wait_for(
            client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": content_parts}],
            ),
            timeout = timeout
        )

        # Safely extract response text
        try:
            if not response.content:
                stop_reason = response.stop_reason
                logger.warning(f"Anthropic response has empty content block for model {model_name}. Stop Reason: {stop_reason}")
                return f"Error: Received empty content block from Anthropic (Stop Reason: {stop_reason})."

            # Combine text from potentially multiple text blocks
            response_text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            if not response_text:
                stop_reason = response.stop_reason
                logger.warning(f"Anthropic response text is empty for model {model_name}. Stop Reason: {stop_reason}")
                return f"Error: Received empty text content from Anthropic response (Stop Reason: {stop_reason})."

            return response_text.strip()
        except AttributeError as e:
             logger.error(f"Error parsing Anthropic response: {e}. Response: {response}")
             return f"Error: Failed to parse Anthropic response structure."


    async def _analyze_with_ollama(
        self,
        client: Union[ollama.AsyncClient, ollama.Client],
        model_name: str,
        prompt: str,
        images: Optional[List[str]],
        temperature: float,
        timeout: int,
        max_tokens: int = 4096,  # Added max_tokens parameter with default value
    ) -> str:
        """Calls the Ollama Generate API (sync or async)."""
        print(f"DEBUG: _analyze_with_ollama called with model={model_name}, has_images={images is not None}")
        
        # Special handling for LLaVA model
        is_llava = "llava" in model_name.lower()
        print(f"DEBUG: Model identified as LLaVA: {is_llava}")
        
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False, # Get full response
            "options": {
                "temperature": temperature,
                 # Add other Ollama options if needed: num_predict for max_tokens etc.
                 "num_predict": max_tokens # Use max_tokens for num_predict
            },
        }
        
        if images:
            print(f"DEBUG: Processing {len(images)} images for Ollama")
            # Remove data URI prefix if present for Ollama
            processed_images = []
            for i, img_b64 in enumerate(images):
                # Skip None or empty images
                if not img_b64:
                    print(f"DEBUG: Image {i} is None or empty, skipping")
                    continue
                
                try:
                    # For base64 strings, remove data URI prefix if present
                    if img_b64 and isinstance(img_b64, str):
                        if img_b64.startswith("data:"):
                            print(f"DEBUG: Image {i} has data URI prefix, removing it")
                            # Handle both formats: data:image/jpeg;base64,XXXX and data:;base64,XXXX
                            if "," in img_b64:
                                img_b64 = img_b64.split(",", 1)[1]
                            else:
                                print(f"DEBUG: Image {i} has unusual data URI format, using as is")
                    
                    # Ensure we have a valid base64 string
                    try:
                        # Validate the base64 string by attempting to decode it
                        base64.b64decode(img_b64)
                        processed_images.append(img_b64)
                        print(f"DEBUG: Image {i} processed successfully")
                    except Exception as b64_err:
                        print(f"DEBUG: Image {i} has invalid base64 encoding: {type(b64_err).__name__}: {b64_err}")
                except Exception as img_err:
                    print(f"DEBUG: Error processing image {i}: {type(img_err).__name__}: {img_err}")
            
            if is_llava:
                # LLaVA in Ollama expects base64 images in a specific format
                if processed_images:
                    print(f"DEBUG: Using first image for LLaVA (base64 encoded), length: {len(processed_images[0])}")
                    # Only use the first image for LLaVA
                    payload["images"] = [processed_images[0]]
                    
                    # Ensure we're using the right field name for images based on Ollama API
                    if hasattr(ollama, "__version__"):
                        ollama_version = ollama.__version__
                        print(f"DEBUG: Detected Ollama SDK version: {ollama_version}")
                        # Check if newer versions use a different field name
                        if ollama_version > "0.1.0":  # Adjust version check as needed
                            print(f"DEBUG: Using 'images' field for newer Ollama SDK")
                        else:
                            print(f"DEBUG: Using 'images' field for older Ollama SDK") 
                else:
                    print(f"DEBUG: No valid images processed for LLaVA")
            else:
                # For non-LLaVA vision models
                if processed_images:
                    payload["images"] = processed_images
                    print(f"DEBUG: Added {len(processed_images)} images to non-LLaVA model payload")
            
            print(f"DEBUG: Final payload has {len(processed_images if processed_images else [])} processed images")
        
        # Detailed logging of the actual payload being sent (excluding image data for brevity)
        debug_payload = payload.copy()
        if "images" in debug_payload:
            img_count = len(debug_payload["images"])
            img_lens = [len(img) for img in debug_payload["images"]]
            debug_payload["images"] = f"[{img_count} images with lengths {img_lens}]"
        print(f"DEBUG: Sending payload to Ollama: {debug_payload}")
        
        try:
            print(f"DEBUG: Executing Ollama call with client type {type(client)}")
            # Execute the call based on client type
            if isinstance(client, ollama.AsyncClient):
                response = await asyncio.wait_for(client.generate(**payload), timeout=timeout)
            elif isinstance(client, ollama.Client):
                # Run the blocking sync call in the thread pool executor
                loop = asyncio.get_running_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, lambda: client.generate(**payload)),
                    timeout=timeout
                )
            else:
                # This case should not happen if client factory works correctly
                raise TypeError(f"Unsupported Ollama client type: {type(client)}")
            
            print(f"DEBUG: Ollama response received, type={type(response)}")
            
            # Handle different response formats based on Ollama SDK version
            result = None
            
            # Handle response from newer Ollama SDK (returns GenerateResponse object)
            if hasattr(response, 'response') and isinstance(getattr(response, 'response', None), str):
                print(f"DEBUG: Processing response as Ollama SDK object")
                result = response.response
                print(f"DEBUG: Extracted response text from Ollama object: {result[:100]}...")
                
                # If the response contains a JSON block within ```json ... ``` markers, extract it
                if "```json" in result:
                    json_start = result.find("```json") + 7  # Length of "```json"
                    json_end = result.find("```", json_start)
                    if json_start != -1 and json_end != -1:
                        json_text = result[json_start:json_end].strip()
                        print(f"DEBUG: Extracted JSON from code block: {json_text[:100]}...")
                        try:
                            # Try to parse JSON to ensure it's valid
                            json.loads(json_text)
                            # If valid, return just the JSON part
                            return json_text
                        except json.JSONDecodeError:
                            print(f"DEBUG: Extracted text was not valid JSON, returning full response")
                            # If not valid JSON, still return the original full response
                            pass
            
            # Handle response from older Ollama SDK (returns dict)
            elif isinstance(response, dict) and "response" in response:
                print(f"DEBUG: Processing response as dictionary")
                result = response.get("response")
                print(f"DEBUG: Response keys: {response.keys()}")
            
            # Handle other potential formats
            elif isinstance(response, str):
                print(f"DEBUG: Response is already a string")
                result = response
            
            # Check if we successfully extracted the result
            if not result:
                logger.warning(f"Ollama response missing 'response' value for model {model_name}. Response: {response}")
                
                # Try to extract error message if available
                error_msg = None
                if isinstance(response, dict) and "error" in response:
                    error_msg = response.get("error")
                elif hasattr(response, 'error') and response.error:
                    error_msg = response.error
                
                if error_msg:
                    print(f"DEBUG: Ollama error: {error_msg}")
                    return f"Error: Ollama returned an error - {error_msg}"
                
                # Fallback: try to use the string representation of the response
                try:
                    print(f"DEBUG: Attempting to use string representation of response")
                    # Extract the JSON part from the response if it looks like a JSON string
                    str_response = str(response)
                    
                    # If the response appears to contain JSON enclosed in ```json ... ```
                    json_start = str_response.find("```json")
                    if json_start != -1:
                        json_start = str_response.find("{", json_start)
                        json_end = str_response.rfind("}")
                        if json_start != -1 and json_end != -1 and json_end > json_start:
                            # Extract the JSON object
                            result = str_response[json_start:json_end+1]
                            print(f"DEBUG: Extracted JSON from markdown code block: {result[:100]}...")
                            return result
                    
                    # If it's a regular response log but not returning it
                    print(f"DEBUG: String representation: {str_response}")
                    
                    # Last resort - use the entire string if we found nothing better
                    if not result and str_response:
                        result = str_response
                        
                except Exception as str_err:
                    print(f"DEBUG: Failed to handle string representation: {str_err}")
                
                if not result:
                    return "Error: Received empty or incomplete response from Ollama."
                
            return result.strip()
            
        except Exception as e:
            print(f"DEBUG: Exception in _analyze_with_ollama: {type(e).__name__}: {e}")
            return f"Error: Ollama API error - {type(e).__name__}: {e}"

    async def close(self):
        """Shuts down the thread pool executor and closes client connections."""
        logger.info("Shutting down AIModelManager resources...")
        await self.client_factory.close_all()
        self._executor.shutdown(wait=True)
        logger.info("AIModelManager resources shut down.")


# --- AI Analysis Module ---
class AIAnalysisModule:
    """Orchestrates AI analysis tasks, including the AI Board feature."""

    def __init__(self):
        self.key_manager = APIKeyManager()
        self.model_registry = ModelRegistry()
        self.model_manager = AIModelManager(self.key_manager, self.model_registry)
        logger.info("AIAnalysisModule initialized.")

    async def run_ai_board_analysis_for_clip(
        self, clip: Dict, project: Dict, board_config: Dict
    ) -> Dict:
        """
        Runs multi-model AI analysis ('AI Board') for a single clip.

        Args:
            clip: The clip data dictionary.
            project: The project data dictionary (for context like settings).
            board_config: Configuration for the AI Board ('board_members', 'tasks', etc.).

        Returns:
            The input clip dictionary updated with AI Board analysis results under keys
            like 'ai_board_raw_results', 'ai_viral_score', 'ai_tags', etc., or error keys.
        """
        if not board_config.get("board_enabled") or not board_config.get("board_members"):
            logger.debug(f"AI Board analysis skipped for clip {clip.get('id', 'N/A')}: Board disabled or no members.")
            return clip # No analysis if board is off or empty

        members = board_config["board_members"] # List of {'provider': str, 'model': str}
        tasks = board_config.get("board_tasks", []) # List of task names (e.g., "viral_analysis")
        chairperson_config = board_config.get("chairperson") # {'provider': str, 'model': str} or None

        clip_id = clip.get('id', 'Unknown')
        logger.info(
            f"Running AI Board analysis for clip {clip_id} with {len(members)} members on tasks: {tasks}."
        )
        logger.debug(f"Board members: {members}")
        logger.debug(f"Chairperson: {chairperson_config}")

        # --- Prepare Shared Inputs ---
        # Encode representative image (thumbnail) if available and needed
        base64_image = None
        thumbnail_path = clip.get("thumbnail")
        logger.debug(f"Thumbnail path: {thumbnail_path}")
        
        # Check if any member or chairperson model *might* use an image
        image_needed = any(
            "image" in self.model_registry.get_model_info(m["provider"], m["model"]).get("capabilities", [])
            for m in members + ([chairperson_config] if chairperson_config else [])
            if self.model_registry.get_model_info(m["provider"], m["model"])
        )
        logger.debug(f"Image needed for analysis: {image_needed}")

        if image_needed and thumbnail_path and os.path.exists(thumbnail_path):
            try:
                project_settings = project.get("settings", {})
                img_quality = int(project_settings.get("compression_quality", 80))
                max_res = int(project_settings.get("max_resolution", 720))
                logger.debug(f"Encoding image with quality={img_quality}, max_resolution={max_res}")
                
                # Use the imported utility function
                base64_image = optimize_and_encode_image(
                    thumbnail_path, quality=img_quality, max_resolution=max_res
                )
                logger.debug(f"Image encoded successfully: {base64_image is not None}")
                if not base64_image:
                    logger.warning(f"Failed to encode thumbnail {thumbnail_path} for clip {clip_id}.")
            except NameError:
                 logger.error("optimize_and_encode_image function not available. Cannot provide image to AI.")
            except Exception as e:
                logger.error(
                    f"Error preparing image for AI board analysis (clip {clip_id}): {e}", exc_info=True
                base64_image = None ) Ensure it's None on error
        elif image_needed and not (thumbnail_path and os.path.exists(thumbnail_path)):
             logger.warning(f"Thumbnail not found or path invalid for clip {clip_id}. Vision models may lack visual context.")

        # Base context prompt
        base_context = (
            f"Context: Analyzing video clip segment (approx time {clip.get('start', 0):.1f}s to {clip.get('end', 0):.1f}s). "
            f"Initial analysis: Score={clip.get('score', 'N/A')}, Tag='{clip.get('tag', 'N/A')}', Category='{clip.get('category', 'N/A')}', Quip='{clip.get('quip', 'N/A')}'. "
            "Consider visual content (if provided) and segment timing. "
        )

        # --- Define Task Prompts ---
        # Structure: { task_name: { prompt_suffix: str, output_keys: List[str], score_key: Optional[str] } }
        task_definitions = {
            "viral_analysis": {
                "prompt_suffix": "Task: Analyze potential virality on social media (TikTok, Reels, Shorts). Provide 'viral_score' (int 0-100), 'viral_tags' (list of keywords/hashtags), and 'viral_recommendations' (brief tips for sharing).",
                "output_keys": ["viral_score", "viral_tags", "viral_recommendations"],
                "score_key": "viral_score",
            },
            "monetization_analysis": {
                "prompt_suffix": "Task: Analyze monetization potential (ads, sponsors, products). Provide 'monetization_score' (int 0-100), 'monetization_tags' (keywords), and 'monetization_recommendations' (brief tips).",
                "output_keys": ["monetization_score", "monetization_tags", "monetization_recommendations"],
                "score_key": "monetization_score",
            },
            # Add more tasks here following the same structure
        }

        # Filter tasks to run based on board_config
        tasks_to_run = {
            task_name: details
            for task_name in tasks
            if task_name in task_definitions
        }
        if not tasks_to_run:
            logger.warning(f"No valid/defined tasks selected for AI Board analysis on clip {clip_id}.")
            clip["ai_board_error"] = "No valid tasks configured."
            return clip

        # --- Define Helper for Running Member Tasks ---
        async def run_single_member_task(member_info: Dict, task_name: str, task_details: Dict) -> Dict:
            """Runs one task for one member, returning a structured result."""
            provider = member_info["provider"]
            model_name = member_info["model"]
            member_key = f"{provider}:{model_name}" # Unique identifier for this member
            task_prompt = f"{base_context}\n\n{task_details['prompt_suffix']}\n\nRespond ONLY with a valid JSON object containing the keys: {task_details['output_keys']}."

            # Determine if this specific model needs the image
            member_images = None
            model_info = self.model_registry.get_model_info(provider, model_name)
            if base64_image and model_info and "image" in model_info.get("capabilities", []):
                 member_images = [base64_image] # Pass as list

            logger.debug(f"Clip {clip_id} - Member {member_key}: Starting task '{task_name}'")

            try:
                response_text = await self.model_manager.analyze_with_model(
                    provider=provider,
                    model_name=model_name,
                    prompt=task_prompt,
                    images=member_images,
                    temperature=0.6, # Slightly creative for analysis tasks
                    max_tokens=DEFAULT_MAX_TOKENS,
                    timeout=DEFAULT_API_TIMEOUT,
                    format_type="json", # Request JSON format
                )

                # Prepare result structure
                task_result = {"member_key": member_key, "task": task_name}

                if response_text.startswith("Error:"):
                    logger.error(f"Clip {clip_id} - Member {member_key} - Task '{task_name}': API Error - {response_text}")
                    task_result.update({"status": "error", "error": response_text})
                    return task_result

                # Attempt to parse and validate JSON response
                try:
                    # Basic JSON cleaning (find first '{' and last '}')
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}")
                    if json_start != -1 and json_end != -1 and json_end > json_start:
                        clean_response = response_text[json_start : json_end + 1]
                        parsed_data = json.loads(clean_response)

                        # Validate required keys
                        missing_keys = [k for k in task_details["output_keys"] if k not in parsed_data]
                        if missing_keys:
                            raise ValueError(f"Missing expected JSON keys: {', '.join(missing_keys)}")

                        # Validate score types if applicable
                        score_key = task_details.get("score_key")
                        if score_key and score_key in parsed_data:
                            try:
                                parsed_data[score_key] = int(parsed_data[score_key])
                                # Clamp score 0-100
                                parsed_data[score_key] = max(0, min(100, parsed_data[score_key]))
                            except (ValueError, TypeError):
                                raise ValueError(f"Invalid type for score key '{score_key}', expected integer.")

                        logger.debug(f"Clip {clip_id} - Member {member_key} - Task '{task_name}': Success")
                        task_result.update({"status": "success", "result": parsed_data})

                    else:
                        raise json.JSONDecodeError("JSON object markers not found", response_text, 0)

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.error(f"Clip {clip_id} - Member {member_key} - Task '{task_name}': JSON/Validation Error - {e}. Raw Response: '{response_text[:250]}...'")
                    task_result.update({"status": "error", "error": f"JSON/Validation Error: {e}", "raw_response": response_text})

                return task_result

            except Exception as e:
                logger.error(f"Clip {clip_id} - Member {member_key} - Task '{task_name}': Unexpected failure - {e}", exc_info=True)
                return {"member_key": member_key, "task": task_name, "status": "error", "error": f"Unexpected error during task: {e}"}


        # --- Execute Member Tasks Concurrently ---
        # WARNING: Running multiple concurrent asyncio tasks within a single ThreadPoolExecutor thread
        #          submitted from Streamlit's main thread can be complex and potentially lead to issues
        #          like deadlocks or loop conflicts if not managed carefully.
        #          Consider alternative architectures if performance/stability issues arise.
        #          For this refactor, we stick to a simpler (potentially blocking) sequential approach.
        #
        # A simple sequential approach:
        updated_project_clips = [] # Collect updated clips
        try:
            for i, clip_data in enumerate(clips):
                clip_id = clip_data.get("id", f"index_{i}")
                status_placeholder.text(f"Analyzing clip {i+1}/{len(clips)} (ID: {clip_id})...")

                # Run the async analysis function for the current clip
                try:
                     # asyncio.run creates its own event loop
                     # This will BLOCK the Streamlit thread until run_ai_board_analysis_for_clip completes
                     updated_clip = asyncio.run(
                          self.run_ai_board_analysis_for_clip(
                              clip_data.copy(), project, board_config
                          )
                     )
                     updated_project_clips.append(updated_clip)

                     # Optional: Save results immediately to DB (if function exists)
                     if update_clip_data(updated_clip["id"], updated_clip):
                         logger.debug(f"Saved AI Board results for clip {updated_clip['id']} to DB.")
                     else:
                         logger.error(f"Failed to save AI Board results for clip {updated_clip['id']} to DB.")
                     processed_count += 1

                     if updated_clip.get("ai_board_status") != "completed":
                          error_count += 1

                except Exception as analysis_exc:
                     logger.error(f"Fatal error running AI Board analysis for clip {clip_id}: {analysis_exc}", exc_info=True)
                     error_count += 1
                     # Add error info to the original clip and append it
                     clip_data["ai_board_error"] = f"Analysis execution failed: {analysis_exc}"
                     clip_data["ai_board_status"] = "execution_error"
                     updated_project_clips.append(clip_data)

                # Update progress bar
                progress = (i + 1) / len(clips)
                progress_bar.progress(progress, text=f"Analyzed {i+1}/{len(clips)} clips...")

            # Update the project dictionary in session state or trigger a reload
            # For now, assume the caller handles project state update based on updated_project_clips
            # A common pattern is to update the database and then st.rerun()

            status_placeholder.success(f"AI Board analysis complete. Processed: {processed_count}, Errors: {error_count}.")
            progress_bar.progress(1.0, text="Analysis Complete!")
            time.sleep(2) # Allow user to see completion message
            progress_bar.empty()
            status_placeholder.empty()

            # Force Streamlit to rerun the script to reflect updated clip data
            # This assumes clip data is reloaded from the source (e.g., DB) at the start of the script
            st.rerun()

        except Exception as e:
             logger.error(f"Error during AI Board batch analysis trigger: {e}", exc_info=True)
             st.error(f"An error occurred during the analysis process: {e}")
             progress_bar.empty() # Clear progress bar on outer error
             status_placeholder.empty()

        # --- Display Results ---
        st.markdown("#### Clip Analysis Results:")

        # Sort clips (e.g., by start time or score) for consistent display
        display_clips = sorted(clips, key=lambda c: c.get("start", 0))

        if not display_clips:
            st.info("No clips available to display results.")
            return

        # Use session state to manage expander states if needed, or rely on Streamlit's default behavior
        # Example without explicit state management (simpler):
        for clip in display_clips:
            clip_id = clip["id"]
            ai_status = clip.get("ai_board_status") # 'completed', 'synthesis_failed', 'analysis_failed', 'execution_error', or None

            # Create expander label based on status
            expander_label = f"Clip @ {clip.get('start', 0):.1f}s - {clip.get('end', 0):.1f}s"
            if tag := clip.get("tag"): expander_label += f": {tag}"

            if ai_status == "completed":
                viral = clip.get('ai_viral_score')
                monet = clip.get('ai_monetization_score')
                score_label = f" (AI Scores: V={viral if viral is not None else 'N/A'}, M={monet if monet is not None else 'N/A'})"
                expander_label += score_label
            elif ai_status:
                 expander_label += f" (Status: {ai_status.replace('_', ' ').title()})"
            else:
                 expander_label += " (Analysis Not Run)"

            # Use a unique key for the expander
            expander_key = f"ai_board_expander_{project_id}_{clip_id}"
            with st.expander(expander_label, expanded=False): # Default to collapsed

                # Display basic clip info using the helper
                self._display_clip_details(clip)
                st.markdown("---")

                # Display AI Board results or status
                if ai_status == "completed":
                    st.markdown("##### AI Board Synthesized Results:")
                    ai_viral = clip.get("ai_viral_score")
                    ai_monetize = clip.get("ai_monetization_score")

                    # Display scores if available
                    score_cols = st.columns(2)
                    with score_cols[0]:
                        st.metric("AI Viral Score", ai_viral if ai_viral is not None else "N/A")
                    with score_cols[1]:
                        st.metric("AI Monetization Score", ai_monetize if ai_monetize is not None else "N/A")

                    # Display tags
                    ai_tags = clip.get("ai_tags", [])
                    if ai_tags:
                        st.markdown("**Consolidated AI Tags:**")
                        # Simple bullet points for tags
                        st.write(", ".join(ai_tags))
                        # Alternative styled tags:
                        # tag_html = " ".join(f"<span style='display:inline-block; background-color:#444; color: #eee; padding: 1px 5px; border-radius:10px; margin-right:3px; margin-bottom:3px; font-size:0.8em;'>{tag}</span>" for tag in ai_tags)
                        # st.markdown(tag_html, unsafe_allow_html=True)

                    # Display recommendations
                    ai_recs = clip.get("ai_recommendations", [])
                    if ai_recs:
                        st.markdown("**Consolidated AI Recommendations:**")
                        for rec in ai_recs:
                            st.markdown(f"- {rec}")

                    # Display summary
                    ai_summary = clip.get("ai_board_summary")
                    if ai_summary:
                        st.markdown("**Chairperson Summary:**")
                        st.markdown(f"> {ai_summary}")

                    # Optionally show raw results in another expander
                    raw_results = clip.get("ai_board_raw_results")
                    if raw_results:
                        with st.expander("View Raw Board Member Results"):
                            st.json(raw_results, expanded=False)

                elif ai_status:
                    # Display errors if analysis failed or had issues
                    st.error(f"AI Board analysis status: {ai_status.replace('_', ' ').title()}")
                    if error_msg := clip.get("ai_board_error"):
                        st.error(f"Board Error: {error_msg}")
                    if synth_error_msg := clip.get("ai_board_synthesis_error"):
                        st.error(f"Synthesis Error: {synth_error_msg}")
                    # Optionally show raw results even on failure
                    raw_results = clip.get("ai_board_raw_results")
                    if raw_results:
                        with st.expander("View Raw Board Member Results (May be incomplete)"):
                            st.json(raw_results, expanded=False)
                else:
                    st.info("AI Board analysis has not been run for this clip yet.")


    async def close(self):
        """Closes resources held by the AI managers."""
        await self.model_manager.close()
        logger.info("AIAnalysisModule resources closed.")


# Example usage pattern (if this script were run directly or imported)
async def main():
    # This is for demonstration; actual use would be within the Streamlit app flow
    logger.info("Initializing AI Analysis Module for demo...")
    ai_module = AIAnalysisModule()

    # Example: List providers with keys
    providers_with_keys = ai_module.key_manager.list_providers_with_keys()
    logger.info(f"Providers with keys found: {providers_with_keys}")

    # Example: Get model info
    model_info = ai_module.model_registry.get_model_info("openai", "gpt-4o")
    logger.info(f"GPT-4o info: {model_info}")

    # Example: Run analysis (requires valid setup and keys)
    # try:
    #     if "openai" in providers_with_keys:
    #         response = await ai_module.model_manager.analyze_with_model(
    #             provider="OpenAI",
    #             model_name="gpt-4o",
    #             prompt="What is the capital of France?",
    #             format_type="text"
    #         )
    #         logger.info(f"Demo OpenAI Response: {response}")
    #     else:
    #         logger.warning("Skipping OpenAI demo analysis: No API key found.")
    # except Exception as e:
    #     logger.error(f"Demo analysis failed: {e}")

    # Close resources when done
    await ai_module.close()

if __name__ == "__main__":
    # Note: Running this directly might have issues with Streamlit session state
    # It's primarily intended to be used within a running Streamlit application.
    # However, this allows testing core non-UI logic if needed.
    # asyncio.run(main())
    logger.info("AI Analysis module script loaded. Run within Streamlit context for full functionality.")

# --- Helper Functions ---
def debug_print(message: str) -> None:
    """Helper function for debug prints that can be easily enabled/disabled."""
    # Set DEBUG_MODE to True to enable debug prints
    DEBUG_MODE = os.environ.get("OPENCLIP_DEBUG", "").lower() in ("true", "1", "yes")
    
    if DEBUG_MODE:
        print(f"DEBUG: {message}")
    
    # Always log at debug level for debugging via log files
    logger.debug(message)

