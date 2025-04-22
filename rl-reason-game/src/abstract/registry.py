from typing import Dict, Type, Any, TypeVar, Generic

T = TypeVar('T')

class BaseRegistry(Generic[T]):
    """
    Base registry class for managing and retrieving implementations.
    
    This class provides a generic implementation that can be used by
    specific registries like GameRegistry and RewardRegistry.
    """
    
    _registry: Dict[str, Type[T]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator for registering an implementation.
        
        Args:
            name: Name to register the implementation under
            
        Returns:
            Decorator function
        """
        def decorator(implementation_class: Type[T]):
            cls._registry[name] = implementation_class
            return implementation_class
        return decorator
    
    @classmethod
    def get_implementation(cls, name: str, config: Dict[str, Any]) -> T:
        """
        Get an instance of a registered implementation.
        
        Args:
            name: Name of the registered implementation
            config: Configuration for the implementation instance
            
        Returns:
            Instance of the requested implementation
            
        Raises:
            ValueError: If the implementation name is not registered
        """
        if name not in cls._registry:
            raise ValueError(f"Implementation '{name}' not found in registry. "
                            f"Available implementations: {list(cls._registry.keys())}")
            
        implementation_class = cls._registry[name]
        return implementation_class(config)
    
    @classmethod
    def list_implementations(cls) -> list:
        """
        List all registered implementations.
        
        Returns:
            List of registered implementation names
        """
        return list(cls._registry.keys())
