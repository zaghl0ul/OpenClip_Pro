"""Data models for type safety and validation."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from ai.error_handling import validate_input


@dataclass
class ClipData:
    """Data model for a video clip."""
    id: UUID = field(default_factory=uuid4)
    project_id: UUID = field(default_factory=uuid4)
    start: float = 0.0
    end: float = 0.0
    score: int = 0
    tag: str = ""
    category: str = "Other"
    quip: str = ""
    thumbnail: Optional[Path] = None
    clip_path: Optional[Path] = None
    
    # AI analysis results
    ai_viral_score: Optional[int] = None
    ai_viral_tags: List[str] = field(default_factory=list)
    ai_viral_recommendations: List[str] = field(default_factory=list)
    ai_monetization_score: Optional[int] = None
    ai_monetization_tags: List[str] = field(default_factory=list)
    ai_monetization_recommendations: List[str] = field(default_factory=list)
    ai_board_summary: Optional[str] = None
    ai_board_status: Optional[str] = None
    
    # Metadata
    duration: Optional[float] = None
    colors: List[str] = field(default_factory=list)
    virality_explanation: Optional[str] = None
    score_explanation: Optional[str] = None
    
    # Raw data storage
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process after initialization."""
        # Convert string UUIDs to UUID objects
        if isinstance(self.id, str):
            self.id = UUID(self.id)
        if isinstance(self.project_id, str):
            self.project_id = UUID(self.project_id)
        
        # Convert paths
        if self.thumbnail and not isinstance(self.thumbnail, Path):
            self.thumbnail = Path(self.thumbnail)
        if self.clip_path and not isinstance(self.clip_path, Path):
            self.clip_path = Path(self.clip_path)
        
        # Calculate duration if not set
        if self.duration is None and self.end > self.start:
            self.duration = self.end - self.start
        
        # Validate
        self.validate()
    
    def validate(self) -> None:
        """Validate clip data."""
        validate_input(self.start, float, "start", min_value=0.0)
        validate_input(self.end, float, "end", min_value=self.start)
        validate_input(self.score, int, "score", min_value=0, max_value=100)
        validate_input(self.tag, str, "tag", min_length=1)
        validate_input(self.category, str, "category", min_length=1)
        
        if self.ai_viral_score is not None:
            validate_input(self.ai_viral_score, int, "ai_viral_score", 
                          min_value=0, max_value=100)
        
        if self.ai_monetization_score is not None:
            validate_input(self.ai_monetization_score, int, "ai_monetization_score",
                          min_value=0, max_value=100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "tag": self.tag,
            "category": self.category,
            "quip": self.quip,
            "thumbnail": str(self.thumbnail) if self.thumbnail else None,
            "clip_path": str(self.clip_path) if self.clip_path else None,
            "duration": self.duration,
            "colors": self.colors,
            "virality_explanation": self.virality_explanation,
            "score_explanation": self.score_explanation,
        }
        
        # Add AI results if present
        if self.ai_viral_score is not None:
            result.update({
                "ai_viral_score": self.ai_viral_score,
                "ai_viral_tags": self.ai_viral_tags,
                "ai_viral_recommendations": self.ai_viral_recommendations,
            })
        
        if self.ai_monetization_score is not None:
            result.update({
                "ai_monetization_score": self.ai_monetization_score,
                "ai_monetization_tags": self.ai_monetization_tags,
                "ai_monetization_recommendations": self.ai_monetization_recommendations,
            })
        
        if self.ai_board_summary:
            result["ai_board_summary"] = self.ai_board_summary
        
        if self.ai_board_status:
            result["ai_board_status"] = self.ai_board_status
        
        # Include any extra data
        result.update(self.data)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClipData":
        """Create from dictionary."""
        # Extract known fields
        known_fields = {
            "id", "project_id", "start", "end", "score", "tag", "category",
            "quip", "thumbnail", "clip_path", "duration", "colors",
            "virality_explanation", "score_explanation",
            "ai_viral_score", "ai_viral_tags", "ai_viral_recommendations",
            "ai_monetization_score", "ai_monetization_tags", 
            "ai_monetization_recommendations", "ai_board_summary", "ai_board_status"
        }
        
        clip_data = {k: v for k, v in data.items() if k in known_fields}
        extra_data = {k: v for k, v in data.items() if k not in known_fields}
        
        # Create instance
        clip = cls(**clip_data)
        clip.data = extra_data
        
        return clip


@dataclass
class ProjectData:
    """Data model for a project."""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    base_dir: Optional[Path] = None
    source_path: Optional[Path] = None
    source_type: str = "file"  # "file" or "youtube"
    
    # Clips
    clips: List[ClipData] = field(default_factory=list)
    
    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    total_clips: int = 0
    analyzed_clips: int = 0
    top_score: int = 0
    average_score: float = 0.0
    duration_seconds: Optional[float] = None
    video_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # AI Board configuration
    ai_board_config: Dict[str, Any] = field(default_factory=dict)
    
    # Extra data
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and process after initialization."""
        # Convert string UUID to UUID object
        if isinstance(self.id, str):
            self.id = UUID(self.id)
        
        # Convert paths
        if self.base_dir and not isinstance(self.base_dir, Path):
            self.base_dir = Path(self.base_dir)
        if self.source_path and not isinstance(self.source_path, Path):
            self.source_path = Path(self.source_path)
        
        # Convert datetime strings
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.updated_at and isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        
        # Update metadata
        self.update_metadata()
        
        # Validate
        self.validate()
    
    def validate(self) -> None:
        """Validate project data."""
        validate_input(self.name, str, "name", min_length=1)
        validate_input(self.source_type, str, "source_type", 
                      choices=["file", "youtube"])
    
    def update_metadata(self) -> None:
        """Update project metadata from clips."""
        self.total_clips = len(self.clips)
        self.analyzed_clips = sum(1 for c in self.clips if c.score > 0)
        
        if self.clips:
            scores = [c.score for c in self.clips if c.score > 0]
            self.top_score = max(scores) if scores else 0
            self.average_score = sum(scores) / len(scores) if scores else 0.0
    
    def add_clip(self, clip: ClipData) -> None:
        """Add a clip to the project."""
        clip.project_id = self.id
        self.clips.append(clip)
        self.update_metadata()
    
    def remove_clip(self, clip_id: Union[str, UUID]) -> bool:
        """Remove a clip by ID."""
        if isinstance(clip_id, str):
            clip_id = UUID(clip_id)
        
        original_count = len(self.clips)
        self.clips = [c for c in self.clips if c.id != clip_id]
        
        if len(self.clips) < original_count:
            self.update_metadata()
            return True
        
        return False
    
    def get_clip(self, clip_id: Union[str, UUID]) -> Optional[ClipData]:
        """Get a clip by ID."""
        if isinstance(clip_id, str):
            clip_id = UUID(clip_id)
        
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result = {
            "id": str(self.id),
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "base_dir": str(self.base_dir) if self.base_dir else None,
            "source_path": str(self.source_path) if self.source_path else None,
            "source_type": self.source_type,
            "total_clips": self.total_clips,
            "analyzed_clips": self.analyzed_clips,
            "top_score": self.top_score,
            "average_score": self.average_score,
            "duration_seconds": self.duration_seconds,
            "video_metadata": self.video_metadata,
            "settings": self.settings,
            "ai_board_config": self.ai_board_config,
            "clips": [clip.to_dict() for clip in self.clips],
        }
        
        # Include extra data
        result.update(self.data)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectData":
        """Create from dictionary."""
        # Extract known fields
        known_fields = {
            "id", "name", "created_at", "updated_at", "base_dir", "source_path",
            "source_type", "total_clips", "analyzed_clips", "top_score",
            "average_score", "duration_seconds", "video_metadata", "settings",
            "ai_board_config"
        }
        
        project_data = {k: v for k, v in data.items() if k in known_fields and k != "clips"}
        extra_data = {k: v for k, v in data.items() if k not in known_fields and k != "clips"}
        
        # Create instance
        project = cls(**project_data)
        project.data = extra_data
        
        # Add clips
        clips_data = data.get("clips", [])
        for clip_dict in clips_data:
            try:
                clip = ClipData.from_dict(clip_dict)
                project.clips.append(clip)
            except Exception as e:
                # Log error but continue
                pass
        
        # Update metadata
        project.update_metadata()
        
        return project


@dataclass
class AIBoardConfig:
    """Configuration for AI Board analysis."""
    board_enabled: bool = False
    board_members: List[Dict[str, str]] = field(default_factory=list)
    board_tasks: List[str] = field(default_factory=list)
    chairperson: Optional[Dict[str, str]] = None
    
    def add_member(self, provider: str, model: str) -> None:
        """Add a board member."""
        self.board_members.append({"provider": provider, "model": model})
    
    def remove_member(self, provider: str, model: str) -> bool:
        """Remove a board member."""
        original_count = len(self.board_members)
        self.board_members = [
            m for m in self.board_members
            if not (m["provider"] == provider and m["model"] == model)
        ]
        return len(self.board_members) < original_count
    
    def clear_members(self) -> None:
        """Clear all board members."""
        self.board_members.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "board_enabled": self.board_enabled,
            "board_members": self.board_members,
            "board_tasks": self.board_tasks,
            "chairperson": self.chairperson,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIBoardConfig":
        """Create from dictionary."""
        return cls(**data) 