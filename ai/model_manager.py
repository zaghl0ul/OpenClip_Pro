# ai/model_manager.py
"""Simplified AI model manager using refactored components."""

import logging
from typing import Dict, List, Optional, Union

from ai.base import (
    AnalysisRequest,
    AnalysisResponse,
    ModelInfo,
    ProviderType,
    APIKeyManagerProtocol,
    ModelRegistryProtocol,
)
from ai.provider_factory import ProviderFactory
from ai.error_handling import handle_errors, validate_input, APIErrorClassifier
from utils.config_manager import get_config

logger = logging.getLogger(__name__)


class AIModelManager:
    """Manages AI model interactions with simplified interface."""
    
    def __init__(
        self,
        key_manager: APIKeyManagerProtocol,
        model_registry: ModelRegistryProtocol
    ):
        """
        Initialize the model manager.
        
        Args:
            key_manager: API key manager
            model_registry: Model registry
        """
        self.key_manager = key_manager
        self.model_registry = model_registry
        self.provider_factory = ProviderFactory(key_manager)
        self.config = get_config()
    
    @handle_errors(reraise=True, error_prefix="Model analysis")
    async def analyze(
        self,
        provider: Union[str, ProviderType],
        model_name: str,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        format_type: str = "text"
    ) -> AnalysisResponse:
        """
        Analyze content with specified model.
        
        Args:
            provider: Provider name or enum
            model_name: Model name
            prompt: Analysis prompt
            images: Optional list of base64 encoded images
            temperature: Sampling temperature (defaults to config)
            max_tokens: Maximum output tokens (defaults to config)
            timeout: Request timeout in seconds (defaults to config)
            format_type: Output format ("text" or "json")
            
        Returns:
            AnalysisResponse object
        """
        # Validate inputs
        validate_input(prompt, str, "prompt", min_length=1)
        validate_input(format_type, str, "format_type", choices=["text", "json"])
        
        # Convert provider to enum if string
        if isinstance(provider, str):
            try:
                provider_type = ProviderType(provider.lower())
            except ValueError:
                raise ValueError(f"Unknown provider: {provider}")
        else:
            provider_type = provider
        
        # Get model info
        model_info = self.model_registry.get_model_info(
            provider_type.value,
            model_name
        )
        
        if not model_info:
            # Create basic model info if not found
            model_info = ModelInfo(
                name=model_name,
                provider=provider_type,
                requires_api_key=(provider_type != ProviderType.OLLAMA)
            )
        
        # Use defaults from config if not provided
        temperature = temperature if temperature is not None else self.config.default_temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.default_max_tokens
        timeout = timeout if timeout is not None else self.config.default_api_timeout
        
        # Create request
        request = AnalysisRequest(
            prompt=prompt,
            model=model_info,
            images=images or [],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            format_type=format_type
        )
        
        # Validate request
        request.validate()
        
        # Get provider
        provider_instance = await self.provider_factory.get_provider(provider_type)
        
        if not provider_instance:
            raise RuntimeError(f"Failed to initialize provider: {provider_type.value}")
        
        # Perform analysis
        response = await provider_instance.analyze(request)
        
        # Log if error
        if response.is_error:
            error_msg = response.get_error_message()
            logger.error(f"Analysis failed: {error_msg}")
        
        return response
    
    async def analyze_simple(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Simple analysis using default provider and model.
        
        Args:
            prompt: Analysis prompt
            images: Optional images
            **kwargs: Additional parameters for analyze()
            
        Returns:
            Response content string
        """
        provider = kwargs.pop("provider", self.config.default_ai_provider)
        model = kwargs.pop("model", self.config.default_ai_model)
        
        response = await self.analyze(
            provider=provider,
            model_name=model,
            prompt=prompt,
            images=images,
            **kwargs
        )
        
        return response.content
    
    @handle_errors(default_return=[], error_prefix="List models")
    async def list_models(
        self,
        provider: Optional[Union[str, ProviderType]] = None,
        vision_only: bool = False
    ) -> List[ModelInfo]:
        """
        List available models.
        
        Args:
            provider: Optional provider to filter by
            vision_only: Whether to return only vision models
            
        Returns:
            List of ModelInfo objects
        """
        if provider:
            # Convert to enum if string
            if isinstance(provider, str):
                try:
                    provider_type = ProviderType(provider.lower())
                except ValueError:
                    return []
            else:
                provider_type = provider
            
            # Get provider instance
            provider_instance = await self.provider_factory.get_provider(provider_type)
            
            if not provider_instance:
                return []
            
            # Get models from provider
            models = await provider_instance.list_available_models()
            
            # Filter vision models if requested
            if vision_only:
                models = [m for m in models if m.is_vision_model]
            
            return models
        
        else:
            # Get models from all available providers
            all_models = []
            
            for provider_type in self.provider_factory.get_available_providers():
                provider_models = await self.list_models(provider_type, vision_only)
                all_models.extend(provider_models)
            
            return all_models
    
    def get_available_providers(self) -> List[str]:
        """
        Get list of available providers.
        
        Returns:
            List of provider names
        """
        providers = self.provider_factory.get_available_providers()
        return [p.value for p in providers]
    
    async def validate_model(
        self,
        provider: Union[str, ProviderType],
        model_name: str
    ) -> bool:
        """
        Check if a model is available.
        
        Args:
            provider: Provider name or enum
            model_name: Model name
            
        Returns:
            True if model is available
        """
        # Convert provider to enum
        if isinstance(provider, str):
            try:
                provider_type = ProviderType(provider.lower())
            except ValueError:
                return False
        else:
            provider_type = provider
        
        # Get provider instance
        provider_instance = await self.provider_factory.get_provider(provider_type)
        
        if not provider_instance:
            return False
        
        # Validate model
        return await provider_instance.validate_model(model_name)
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.provider_factory.close_all()


# Convenience functions
async def analyze_with_ai(
    prompt: str,
    images: Optional[List[str]] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str:
    """
    Simple function to analyze content with AI.
    
    This is a convenience function that uses the default model manager.
    For more control, use AIModelManager directly.
    
    Args:
        prompt: Analysis prompt
        images: Optional images
        provider: Optional provider (defaults to config)
        model: Optional model (defaults to config)
        **kwargs: Additional parameters
        
    Returns:
        Response content string
    """
    # This would need to be initialized with proper dependencies
    # In practice, this would use a singleton or be initialized at app startup
    raise NotImplementedError(
        "This convenience function requires proper initialization. "
        "Use AIModelManager directly instead."
    ) 