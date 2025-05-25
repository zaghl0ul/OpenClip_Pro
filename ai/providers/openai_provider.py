"""OpenAI provider implementation."""

import base64
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

# Try to import OpenAI SDK
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available. Install with: pip install openai")


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider implementation."""
    
    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.OPENAI
    
    async def _create_client(self) -> None:
        """Create OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed")
        
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            timeout=120.0  # Default timeout
        )
    
    @handle_errors(reraise=True, error_prefix="OpenAI analysis")
    async def analyze(self, request: AnalysisRequest) -> AnalysisResponse:
        """Perform analysis using OpenAI."""
        # Validate request
        request.validate()
        
        if not self._initialized:
            await self.initialize()
        
        # Prepare messages
        messages = self._prepare_messages(request)
        
        # Prepare request parameters
        params = {
            "model": request.model.name,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        
        # Add JSON mode if requested and supported
        if request.format_type == "json" and request.model.supports_json:
            params["response_format"] = {"type": "json_object"}
        
        try:
            # Make API call with timeout
            response = await with_timeout(
                self._client.chat.completions.create(**params),
                timeout=request.timeout,
                error_message=f"OpenAI API call timed out for model {request.model.name}"
            )
            
            # Extract response
            content = response.choices[0].message.content
            
            if content is None:
                finish_reason = response.choices[0].finish_reason
                raise ValueError(f"Received empty response (finish reason: {finish_reason})")
            
            # Build response
            usage = None
            if hasattr(response, 'usage'):
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            return AnalysisResponse(
                content=content.strip(),
                model=request.model,
                usage=usage,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            
            # Classify error for better user feedback
            user_message = APIErrorClassifier.get_user_friendly_message(e)
            return create_error_response(
                Exception(f"{user_message} (Details: {str(e)})"),
                request.model
            )
    
    def _prepare_messages(self, request: AnalysisRequest) -> List[Dict[str, Any]]:
        """Prepare messages for OpenAI API."""
        content_parts = [{"type": "text", "text": request.prompt}]
        
        # Add images if provided
        if request.images:
            for img_b64 in request.images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                })
        
        return [{"role": "user", "content": content_parts}]
    
    async def list_available_models(self) -> List[ModelInfo]:
        """List available OpenAI models."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Fetch models from API
            models_response = await self._client.models.list()
            models = models_response.data
            
            # Filter and convert to ModelInfo
            vision_models = []
            
            for model in models:
                model_id = model.id
                
                # Filter for vision-capable models
                if self._is_vision_model(model_id):
                    model_info = self._create_model_info(model_id)
                    if model_info:
                        vision_models.append(model_info)
            
            return vision_models
            
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return self._get_static_models()
    
    def _is_vision_model(self, model_id: str) -> bool:
        """Check if a model supports vision."""
        vision_prefixes = [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-4-vision",
            "chatgpt-4o",
            "gpt-4-1106-vision",
        ]
        
        return any(model_id.startswith(prefix) for prefix in vision_prefixes)
    
    def _create_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Create ModelInfo from model ID."""
        # Determine quality
        quality = ModelQuality.STANDARD
        if "4o" in model_id or "turbo" in model_id:
            quality = ModelQuality.PREMIUM
        elif "mini" in model_id:
            quality = ModelQuality.FAST
        
        # Determine capabilities
        capabilities = [ModelCapability.TEXT, ModelCapability.JSON]
        if self._is_vision_model(model_id):
            capabilities.append(ModelCapability.IMAGE)
        
        return ModelInfo(
            name=model_id,
            provider=ProviderType.OPENAI,
            capabilities=capabilities,
            quality=quality,
            requires_api_key=True,
            is_vision_model=ModelCapability.IMAGE in capabilities
        )
    
    def _get_static_models(self) -> List[ModelInfo]:
        """Get static list of known models."""
        return [
            ModelInfo(
                name="gpt-4o",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.JSON],
                quality=ModelQuality.PREMIUM,
                requires_api_key=True,
                is_vision_model=True
            ),
            ModelInfo(
                name="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.JSON],
                quality=ModelQuality.FAST,
                requires_api_key=True,
                is_vision_model=True
            ),
            ModelInfo(
                name="gpt-4-turbo",
                provider=ProviderType.OPENAI,
                capabilities=[ModelCapability.TEXT, ModelCapability.IMAGE, ModelCapability.JSON],
                quality=ModelQuality.PREMIUM,
                requires_api_key=True,
                is_vision_model=True
            ),
        ]
    
    async def validate_model(self, model_name: str) -> bool:
        """Check if a model is available."""
        models = await self.list_available_models()
        return any(model.name == model_name for model in models) 