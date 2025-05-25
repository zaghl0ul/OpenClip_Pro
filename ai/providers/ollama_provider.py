"""Ollama provider implementation."""

import base64
import json
import logging
from typing import Any, Dict, List, Optional

from ai.base import (
    AnalysisRequest,
    AnalysisResponse,
    BaseAIProvider,
    ModelCapability,
    ModelInfo,
    ModelQuality,
    ProviderType,
    create_error_response,
    with_timeout,
)
from ai.error_handling import handle_errors, APIErrorClassifier

logger = logging.getLogger(__name__)

# Try to import Ollama SDK
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama SDK not available. Install with: pip install ollama")


class OllamaProvider(BaseAIProvider):
    """Ollama local model provider implementation."""
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OLLAMA
    
    @property
    def requires_api_key(self) -> bool:
        """Ollama runs locally and doesn't require an API key."""
        return False
    
    async def _create_client(self) -> None:
        """Create Ollama client."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama SDK not installed")
        
        try:
            # Prefer AsyncClient if available
            if hasattr(ollama, "AsyncClient"):
                self._client = ollama.AsyncClient(
                    timeout=120.0  # Default timeout
                )
                # Test connection
                await self._client.list()
                logger.info("Ollama AsyncClient connected successfully")
            else:
                # Fallback to sync client
                self._client = ollama.Client(
                    timeout=120.0
                )
                # Test connection (blocking)
                self._client.list()
                logger.warning("Using synchronous Ollama client")
                logger.info("Ollama Client connected successfully")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise ConnectionError(
                f"Could not connect to Ollama server. Ensure it's running. Error: {e}"
            )
    
    @handle_errors(reraise=True, error_prefix="Ollama analysis")
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform analysis using Ollama."""
        # Validate request
        request.validate()
        
        if not self._initialized:
            await self.initialize()
        
        # Prepare payload
        payload = self._prepare_payload(request)
        
        try:
            # Make API call based on client type
            if hasattr(self._client, "generate") and hasattr(self._client, "__aenter__"):
                # Async client
                response = await with_timeout(
                    self._client.generate(**payload),
                    timeout=request.timeout,
                    error_message=f"Ollama API call timed out for model {request.model.name}"
                )
            else:
                # Sync client - run in executor
                import asyncio
                loop = asyncio.get_running_loop()
                response = await with_timeout(
                    loop.run_in_executor(None, lambda: self._client.generate(**payload)),
                    timeout=request.timeout,
                    error_message=f"Ollama API call timed out for model {request.model.name}"
                )
            
            # Extract response content
            content = self._extract_response_content(response)
            
            if not content:
                raise ValueError("Received empty response from Ollama")
            
            # Build response
            return AnalysisResponse(
                content=content.strip(),
                model=request.model,
                finish_reason="complete"
            )
            
        except Exception as e:
            logger.error(f"Ollama API error: {e}", exc_info=True)
            
            # Check if it's a model not found error
            if "model" in str(e).lower() and "not found" in str(e).lower():
                model_name = request.model.name
                return create_error_response(
                    Exception(
                        f"Model '{model_name}' not found. "
                        f"Install it with: ollama pull {model_name}"
                    ),
                    request.model
                )
            
            # Generic error handling
            user_message = APIErrorClassifier.get_user_friendly_message(e)
            return create_error_response(
                Exception(f"{user_message} (Details: {str(e)})"),
                request.model
            )
    
    def _prepare_payload(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Prepare request payload for Ollama API."""
        payload = {
            "model": request.model.name,
            "prompt": request.prompt,
            "stream": False,  # Get full response
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,  # Ollama uses num_predict
            }
        }
        
        # Add images if provided and model supports them
        if request.images and request.model.supports_images:
            # Ollama expects base64 images without data URI prefix
            processed_images = []
            
            for img_b64 in request.images:
                # Remove data URI prefix if present
                if img_b64.startswith("data:"):
                    # Extract base64 part after comma
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                
                processed_images.append(img_b64)
            
            # Ollama expects images in the 'images' field
            payload["images"] = processed_images
        
        return payload
    
    def _extract_response_content(self, response: Any) -> str:
        """Extract content from Ollama response."""
        # Handle different response formats
        if hasattr(response, 'response') and isinstance(response.response, str):
            # Response object with 'response' attribute
            return response.response
        elif isinstance(response, dict) and "response" in response:
            # Dictionary response
            return response["response"]
        elif isinstance(response, str):
            # Direct string response
            return response
        else:
            # Try to extract any reasonable content
            logger.warning(f"Unexpected Ollama response format: {type(response)}")
            return str(response)
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available Ollama models."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get list of models
            if hasattr(self._client, "list"):
                if hasattr(self._client, "__aenter__"):
                    # Async client
                    models_response = await self._client.list()
                else:
                    # Sync client
                    import asyncio
                    loop = asyncio.get_running_loop()
                    models_response = await loop.run_in_executor(
                        None, self._client.list
                    )
            else:
                logger.warning("Ollama client doesn't support listing models")
                return self._get_static_models()
            
            # Parse response
            models = []
            
            # Handle different response formats
            if isinstance(models_response, dict) and "models" in models_response:
                model_list = models_response["models"]
            elif isinstance(models_response, list):
                model_list = models_response
            else:
                logger.warning(f"Unexpected models response format: {type(models_response)}")
                return self._get_static_models()
            
            # Convert to ModelInfo
            for model_data in model_list:
                if isinstance(model_data, dict) and "name" in model_data:
                    model_name = model_data["name"]
                    model_info = self._create_model_info(model_name)
                    if model_info:
                        models.append(model_info)
            
            return models if models else self._get_static_models()
            
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return self._get_static_models()
    
    def _create_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Create ModelInfo from model name."""
        # Clean model name (remove tag if present)
        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        
        # Determine capabilities based on known models
        capabilities = [ModelCapability.TEXT]
        is_vision = False
        
        # Vision models
        vision_models = ["llava", "bakllava", "llava-llama3", "llava-phi3"]
        if any(vm in base_name.lower() for vm in vision_models):
            capabilities.append(ModelCapability.IMAGE)
            is_vision = True
        
        # Determine quality
        quality = ModelQuality.STANDARD
        if "llama3" in base_name.lower() or "mixtral" in base_name.lower():
            quality = ModelQuality.PREMIUM
        elif "phi" in base_name.lower() or "gemma" in base_name.lower():
            quality = ModelQuality.FAST
        
        return ModelInfo(
            name=model_name,
            provider=ProviderType.OLLAMA,
            capabilities=capabilities,
            quality=quality,
            requires_api_key=False,
            is_vision_model=is_vision,
            is_local=True
        )
    
    def _get_static_models(self) -> List[ModelInfo]:
        """Get static list of common Ollama models."""
        return [
            ModelInfo(
                name="llava",
                provider=ProviderType.OLLAMA,
                capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE],
                quality=ModelQuality.STANDARD,
                requires_api_key=False,
                is_vision_model=True,
                is_local=True
            ),
            ModelInfo(
                name="llava:latest",
                provider=ProviderType.OLLAMA,
                capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE],
                quality=ModelQuality.STANDARD,
                requires_api_key=False,
                is_vision_model=True,
                is_local=True
            ),
            ModelInfo(
                name="llama3",
                provider=ProviderType.OLLAMA,
                capabilities=[ModelCapability.TEXT],
                quality=ModelQuality.PREMIUM,
                requires_api_key=False,
                is_vision_model=False,
                is_local=True
            ),
            ModelInfo(
                name="mistral",
                provider=ProviderType.OLLAMA,
                capabilities=[ModelCapability.TEXT],
                quality=ModelQuality.STANDARD,
                requires_api_key=False,
                is_vision_model=False,
                is_local=True
            ),
        ]
    
    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get list of available models
            models = await self.list_available_models()
            model_names = [m.name for m in models]
            
            # Check exact match
            if model_name in model_names:
                return True
            
            # Check base name match (without tag)
            base_name = model_name.split(":")[0] if ":" in model_name else model_name
            for available_name in model_names:
                available_base = available_name.split(":")[0] if ":" in available_name else available_name
                if base_name == available_base:
                    return True
            
            # Special handling for llava
            if "llava" in model_name.lower():
                return any("llava" in m.lower() for m in model_names)
            
            return False
            
        except Exception as e:
            logger.warning(f"Error validating Ollama model: {e}")
            # Assume model might be available if we can't check
            return True 