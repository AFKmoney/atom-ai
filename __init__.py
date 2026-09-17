"""
Toroidal Fractal Intelligence — A continuous structured learning architecture.

This package implements a novel AI architecture based on toroidal fractal
dynamics, where intelligence emerges from the continuous transformation of
tokens into structured atoms that interact, aggregate, abstract, and consolidate.

Key features:
- Token → Toroidal Atom conversion
- Fractal superposition state
- RK4 dynamics integration
- Hierarchical aggregation
- Dynamic abstraction
- Continuous consolidation
- Infinite trainability

Usage:
    from toroidal_fractal_intelligence import create_model, train, chat

    model = create_model()
    result = train(model, max_steps=10000)
    response = chat(model, "Once upon a time")
"""

if __package__:
    from .src.toroidal.model import ToroidalFractalIntelligence
    from .src.main import create_model, train, chat, interactive_chat
else:
    from src.toroidal.model import ToroidalFractalIntelligence
    from src.main import create_model, train, chat, interactive_chat

__version__ = "0.1.0"
__all__ = [
    "ToroidalFractalIntelligence",
    "create_model",
    "train",
    "chat",
    "interactive_chat",
]
