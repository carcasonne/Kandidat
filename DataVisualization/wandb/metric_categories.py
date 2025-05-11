from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

class MetricCategory(Enum):
    """Categories of metrics for different visualization approaches"""
    SCALAR = auto()      # Simple scalar metrics like accuracy, loss, etc.
    DISTRIBUTION = auto() # Metrics that represent distributions
    MATRIX = auto()      # Confusion matrices and other matrix-based metrics
    PLOT = auto()        # Metrics that are already plots (like TSNE)
    
@dataclass
class MetricInfo:
    """Information about a metric including how to display it"""
    name: str
    display_name: str
    category: MetricCategory
    preferred_color: Optional[str] = None
    preferred_style: Optional[str] = None
    description: Optional[str] = None
    
    @property
    def id(self) -> str:
        """A clean identifier for the metric"""
        return self.name.lower().replace(" ", "_")

# Create metric registry
class MetricRegistry:
    """Registry of all available metrics with their metadata"""
    
    def __init__(self):
        self._metrics: Dict[str, MetricInfo] = {}
        
    def register(self, metric: MetricInfo) -> None:
        """Register a new metric"""
        self._metrics[metric.name] = metric
        
    def get(self, name: str) -> MetricInfo:
        """Get a metric by name"""
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self._metrics[name]
    
    def get_by_category(self, category: MetricCategory) -> List[MetricInfo]:
        """Get all metrics of a specific category"""
        return [m for m in self._metrics.values() if m.category == category]
    
    def get_all(self) -> List[MetricInfo]:
        """Get all registered metrics"""
        return list(self._metrics.values())
    
    def __contains__(self, name: str) -> bool:
        return name in self._metrics

# Create global registry
metrics = MetricRegistry()

# Register training metrics
metrics.register(MetricInfo(
    name="Train Accuracy", 
    display_name="Training Accuracy",
    category=MetricCategory.SCALAR,
    preferred_color="blue",
    description="Accuracy on training data"
))

metrics.register(MetricInfo(
    name="Train Loss", 
    display_name="Training Loss",
    category=MetricCategory.SCALAR,
    preferred_color="red",
    description="Loss on training data"
))

# stupid xD
metrics.register(MetricInfo(
    name="Loss",
    display_name="Training Loss",
    category=MetricCategory.SCALAR,
    preferred_color="red",
    description="Loss on training data"
))

metrics.register(MetricInfo(
    name="Train Precision", 
    display_name="Training Precision",
    category=MetricCategory.SCALAR,
    preferred_color="green",
    description="Precision on training data"
))

metrics.register(MetricInfo(
    name="Train Recall", 
    display_name="Training Recall",
    category=MetricCategory.SCALAR,
    preferred_color="purple",
    description="Recall on training data"
))

metrics.register(MetricInfo(
    name="Train F1 Score", 
    display_name="Training F1 Score",
    category=MetricCategory.SCALAR,
    preferred_color="orange",
    description="F1 Score on training data"
))

metrics.register(MetricInfo(
    name="train_conf_mat", 
    display_name="Training Confusion Matrix",
    category=MetricCategory.MATRIX,
    description="Confusion matrix on training data"
))

metrics.register(MetricInfo(
    name="Accuracy", 
    display_name="Accuracy",
    category=MetricCategory.SCALAR,
    preferred_color="blue",
    description="General accuracy metric"
))

# Register validation metrics
metrics.register(MetricInfo(
    name="Val Accuracy", 
    display_name="Validation Accuracy",
    category=MetricCategory.SCALAR,
    preferred_color="blue",
    preferred_style="dashed",
    description="Accuracy on validation data"
))

metrics.register(MetricInfo(
    name="Val Loss", 
    display_name="Validation Loss",
    category=MetricCategory.SCALAR,
    preferred_color="red",
    preferred_style="dashed",
    description="Loss on validation data"
))

metrics.register(MetricInfo(
    name="Val Precision", 
    display_name="Validation Precision",
    category=MetricCategory.SCALAR,
    preferred_color="green",
    preferred_style="dashed",
    description="Precision on validation data"
))

metrics.register(MetricInfo(
    name="Val Recall", 
    display_name="Validation Recall",
    category=MetricCategory.SCALAR,
    preferred_color="purple",
    preferred_style="dashed",
    description="Recall on validation data"
))

metrics.register(MetricInfo(
    name="Val F1 Score", 
    display_name="Validation F1 Score",
    category=MetricCategory.SCALAR,
    preferred_color="orange",
    preferred_style="dashed",
    description="F1 Score on validation data"
))

metrics.register(MetricInfo(
    name="val_conf_mat", 
    display_name="Validation Confusion Matrix",
    category=MetricCategory.MATRIX,
    description="Confusion matrix on validation data"
))

# Special metrics
metrics.register(MetricInfo(
    name="Train Spider Plot", 
    display_name="Training Spider Plot",
    category=MetricCategory.PLOT,
    description="Spider plot of training metrics"
))

metrics.register(MetricInfo(
    name="Val Spider Plot", 
    display_name="Validation Spider Plot",
    category=MetricCategory.PLOT,
    description="Spider plot of validation metrics"
))

# Function to get a metric by name (convenience function)
def get_metric(name: str) -> MetricInfo:
    """Get a metric by name from the global registry"""
    return metrics.get(name)
