"""Example usage of the refactored AI providers."""

import asyncio
import os
from typing import Optional

from ai.model_manager import AIModelManager
from ai.base import APIKeyManagerProtocol, ModelRegistryProtocol, ProviderType
from utils import setup_logging, get_logger, get_config

# Setup logging
setup_logging(console=True, use_colors=True)
logger = get_logger(__name__)


class SimpleKeyManager:
    """Simple API key manager for demonstration."""
    
    def __init__(self):
        self.keys = {}
        # Load from environment
        if api_key := os.environ.get("OPENAI_API_KEY"):
            self.keys["openai"] = api_key
        if api_key := os.environ.get("ANTHROPIC_API_KEY"):
            self.keys["anthropic"] = api_key
        if api_key := os.environ.get("GOOGLE_API_KEY"):
            self.keys["google"] = api_key
    
    def get_key(self, provider: str) -> Optional[str]:
        return self.keys.get(provider.lower())
    
    def save_key(self, provider: str, key: str) -> bool:
        self.keys[provider.lower()] = key
        return True
    
    def delete_key(self, provider: str) -> bool:
        self.keys.pop(provider.lower(), None)
        return True


class SimpleModelRegistry:
    """Simple model registry for demonstration."""
    
    def get_model_info(self, provider: str, model_name: str) -> Optional[dict]:
        # Let providers handle their own model info
        return None
    
    def list_models_for_provider(self, provider: str) -> list[str]:
        # Let providers handle their own model lists
        return []
    
    def list_vision_models(self) -> list[dict]:
        # Let providers handle their own model lists
        return []


async def example_basic_usage():
    """Basic usage example."""
    logger.info("=== Basic Usage Example ===")
    
    # Create manager
    key_manager = SimpleKeyManager()
    model_registry = SimpleModelRegistry()
    manager = AIModelManager(key_manager, model_registry)
    
    try:
        # Example 1: Simple text analysis with Ollama (no API key needed)
        logger.info("\n1. Testing Ollama (local model)...")
        try:
            response = await manager.analyze_simple(
                prompt="What is the meaning of life? Answer in one sentence.",
                provider="ollama",
                model="llava",  # Assuming llava is installed
                temperature=0.7,
                max_tokens=100
            )
            logger.info(f"Ollama response: {response}")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        
        # Example 2: OpenAI analysis (if API key is set)
        if key_manager.get_key("openai"):
            logger.info("\n2. Testing OpenAI...")
            response = await manager.analyze_simple(
                prompt="Write a haiku about programming.",
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.8,
                max_tokens=50
            )
            logger.info(f"OpenAI response:\n{response}")
        else:
            logger.info("\n2. Skipping OpenAI (no API key)")
        
        # Example 3: List available providers
        logger.info("\n3. Available providers:")
        providers = manager.get_available_providers()
        for provider in providers:
            logger.info(f"  - {provider}")
        
    finally:
        await manager.close()


async def example_advanced_usage():
    """Advanced usage example with error handling and model selection."""
    logger.info("\n=== Advanced Usage Example ===")
    
    # Create manager
    key_manager = SimpleKeyManager()
    model_registry = SimpleModelRegistry()
    manager = AIModelManager(key_manager, model_registry)
    
    try:
        # Example 1: List and select models
        logger.info("\n1. Listing available models...")
        
        for provider_name in ["ollama", "openai"]:
            logger.info(f"\n{provider_name.upper()} Models:")
            try:
                models = await manager.list_models(provider_name, vision_only=True)
                if models:
                    for model in models[:3]:  # Show first 3
                        logger.info(f"  - {model.name} (Quality: {model.quality.value})")
                else:
                    logger.info("  No models available")
            except Exception as e:
                logger.info(f"  Error listing models: {e}")
        
        # Example 2: Analyze with specific response handling
        logger.info("\n2. Testing with response object...")
        
        if key_manager.get_key("openai"):
            response = await manager.analyze(
                provider="openai",
                model_name="gpt-4o-mini",
                prompt="Explain quantum computing in simple terms.",
                temperature=0.5,
                max_tokens=150
            )
            
            if response.is_error:
                logger.error(f"Analysis failed: {response.get_error_message()}")
            else:
                logger.info(f"Success! Response: {response.content[:100]}...")
                if response.usage:
                    logger.info(f"Token usage: {response.usage}")
        
        # Example 3: JSON mode
        logger.info("\n3. Testing JSON mode...")
        
        if key_manager.get_key("openai"):
            response = await manager.analyze(
                provider="openai",
                model_name="gpt-4o-mini",
                prompt='Create a JSON object with fields: "name" (a random name) and "age" (a random age).',
                format_type="json",
                temperature=0.9,
                max_tokens=50
            )
            
            if not response.is_error:
                logger.info(f"JSON response: {response.content}")
                # Parse and validate JSON
                try:
                    import json
                    data = json.loads(response.content)
                    logger.info(f"Parsed data: Name={data.get('name')}, Age={data.get('age')}")
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON response")
        
    finally:
        await manager.close()


async def example_vision_analysis():
    """Example of vision analysis with image."""
    logger.info("\n=== Vision Analysis Example ===")
    
    # Check if we have a test image
    from pathlib import Path
    test_image = Path("test_image.jpg")
    
    if not test_image.exists():
        logger.info("No test image found. Creating a simple test image...")
        # Create a simple test image
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.new('RGB', (400, 200), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((50, 80), "Hello AI Vision!", fill='black', font=None)
            draw.rectangle([20, 20, 380, 180], outline='blue', width=3)
            img.save(test_image)
            logger.info("Created test_image.jpg")
        except ImportError:
            logger.warning("PIL not available. Skipping vision example.")
            return
    
    # Create manager
    key_manager = SimpleKeyManager()
    model_registry = SimpleModelRegistry()
    manager = AIModelManager(key_manager, model_registry)
    
    try:
        # Load and encode image
        import base64
        with open(test_image, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Try with different providers
        providers_to_test = []
        
        if key_manager.get_key("openai"):
            providers_to_test.append(("openai", "gpt-4o-mini"))
        
        # Always try Ollama (no key needed)
        providers_to_test.append(("ollama", "llava"))
        
        for provider, model in providers_to_test:
            logger.info(f"\nTesting vision with {provider}/{model}...")
            
            try:
                response = await manager.analyze(
                    provider=provider,
                    model_name=model,
                    prompt="Describe what you see in this image.",
                    images=[image_data],
                    temperature=0.5,
                    max_tokens=100
                )
                
                if response.is_error:
                    logger.warning(f"{provider} error: {response.get_error_message()}")
                else:
                    logger.info(f"{provider} description: {response.content}")
                    
            except Exception as e:
                logger.error(f"Failed with {provider}: {e}")
        
    finally:
        await manager.close()
        # Clean up test image if we created it
        if test_image.exists() and test_image.stat().st_size < 10000:  # Small test image
            test_image.unlink()
            logger.info("Cleaned up test image")


async def main():
    """Run all examples."""
    config = get_config()
    logger.info(f"OpenClip Pro Refactored AI Examples (v{config.app_version})")
    logger.info(f"Debug mode: {config.debug_mode}")
    
    await example_basic_usage()
    await example_advanced_usage()
    await example_vision_analysis()
    
    logger.info("\n✅ All examples completed!")


if __name__ == "__main__":
    asyncio.run(main()) 