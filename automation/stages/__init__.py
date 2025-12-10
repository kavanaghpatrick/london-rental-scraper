"""
Pipeline stages.

Each stage is a self-contained unit that:
1. Takes a config and run_id
2. Returns a StageResult with status and metrics
3. Can be run independently for debugging
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    WARNING = "warning"  # Completed with issues
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of running a pipeline stage."""
    stage_name: str
    status: StageStatus
    started_at: datetime
    finished_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'stage_name': self.stage_name,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'finished_at': self.finished_at.isoformat() if self.finished_at else None,
            'duration_seconds': self.duration_seconds,
            'items_processed': self.items_processed,
            'items_failed': self.items_failed,
            'error_message': self.error_message,
            'warnings': self.warnings,
            'metrics': self.metrics,
        }
