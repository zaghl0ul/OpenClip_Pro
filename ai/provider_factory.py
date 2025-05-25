# ai/provider_factory.py
"""Factory for creating and managing AI providers."""

import logging
from typing import Dict, Optional, Type

from ai.base import AIProvider, ProviderType, APIKeyManagerProtocol
from ai.error_handling import handle_errors

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating and caching AI provider instances."""
    
    def __init__(self, key_manager: APIKeyManagerProtocol):
        """
        Initialize the factory.
        
        Args:
            key_manager: API key manager instance
        """
        self.key_manager = key_manager
        self._providers: Dict[ProviderType, AIProvider] = {}
        self._provider_classes: Dict[ProviderType, Type[AIProvider]] = {}
        
        # Register default providers
        self._register_default_providers()
    
    def _register_default_providers(self) -> None:
        """Register default provider implementations."""
        # Import providers here to avoid circular imports
        try:
            from ai.providers.openai_provider import OpenAIProvider
            self.register_provider(ProviderType.OPENAI, OpenAIProvider)
        except ImportError:
            logger.warning("OpenAI provider not available")
        
        try:
            from ai.providers.anthropic_provider import AnthropicProvider
            self.register_provider(ProviderType.ANTHROPIC, AnthropicProvider)
        except ImportError:
            logger.warning("Anthropic provider not available")
        
        try:
            from ai.providers.google_provider import GoogleProvider
            self.register_provider(ProviderType.GOOGLE, GoogleProvider)
        except ImportError:
            logger.warning("Google provider not available")
        
        try:
            from ai.providers.ollama_provider import OllamaProvider
            self.register_provider(ProviderType.OLLAMA, OllamaProvider)
        except ImportError:
            logger.warning("Ollama provider not available")
    
    def register_provider(
        self,
        provider_type: ProviderType,
        provider_class: Type[AIProvider]
    ) -> None:
        """
        Register a provider class.
        
        Args:
            provider_type: Provider type enum
            provider_class: Provider class to register
        """
        self._provider_classes[provider_type] = provider_class
        logger.debug(f"Registered provider: {provider_type.value}")
    
    @handle_errors(default_return=None, error_prefix="Provider creation")
    async def get_provider(
        self,
        provider_type: ProviderType,
        force_new: bool = False
    ) -> Optional[AIProvider]:
        """
        Get or create a provider instance.
        
        Args:
            provider_type: Provider type to get
            force_new: Force creation of new instance
            
        Returns:
            Provider instance or None if creation fails
        """
        # Return cached instance if available
        if not force_new and provider_type in self._providers:
            return self._providers[provider_type]
        
        # Check if provider class is registered
        if provider_type not in self._provider_classes:
            logger.error(f"Provider not registered: {provider_type.value}")
            return None
        
        # Get API key if required
        api_key = None
        if provider_type != ProviderType.OLLAMA:
            api_key = self.key_manager.get_key(provider_type.value)
            if not api_key:
                logger.error(f"No API key found for provider: {provider_type.value}")
                return None
        
        # Create provider instance
        provider_class = self._provider_classes[provider_type]
        provider = provider_class(api_key=api_key)
        
        # Initialize provider
        try:
            await provider.initialize()
            self._providers[provider_type] = provider
            logger.info(f"Created provider instance: {provider_type.value}")
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider {provider_type.value}: {e}")
            return None
    
    async def close_all(self) -> None:
        """Close all provider instances."""
        for provider_type, provider in self._providers.items():
            try:
                await provider.close()
                logger.debug(f"Closed provider: {provider_type.value}")
            except Exception as e:
                logger.error(f"Error closing provider {provider_type.value}: {e}")
        
        self._providers.clear()
    
    def get_available_providers(self) -> list[ProviderType]:
        """Get list of available providers."""
        available = []
        
        for provider_type in self._provider_classes:
            # Check if provider has API key (except Ollama)
            if provider_type == ProviderType.OLLAMA:
                available.append(provider_type)
            else:
                api_key = self.key_manager.get_key(provider_type.value)
                if api_key:
                    available.append(provider_type)
        
        return available 