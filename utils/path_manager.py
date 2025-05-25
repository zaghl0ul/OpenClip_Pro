"""Centralized path management utilities."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from ai.error_handling import handle_errors, validate_input


class PathManager:
    """Centralized path management for the application."""
    
    # Standard subdirectories for projects
    PROJECT_SUBDIRS = [
        "clips",
        "thumbnails",
        "exports",
        "temp",
        "frames",
    ]
    
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initialize PathManager.
        
        Args:
            base_dir: Base directory for the application
        """
        self.base_dir = Path(base_dir) if base_dir else Path.home() / ".openclip"
        self._ensure_base_dir()
    
    @handle_errors(reraise=True, error_prefix="Path creation")
    def _ensure_base_dir(self) -> None:
        """Ensure base directory exists."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    @handle_errors(reraise=True, error_prefix="Project directory creation")
    def create_project_directories(self, project_id: str) -> Dict[str, Path]:
        """
        Create all required directories for a project.
        
        Args:
            project_id: Unique project identifier
            
        Returns:
            Dictionary mapping directory names to Path objects
        """
        validate_input(project_id, str, "project_id", min_length=1)
        
        project_base = self.base_dir / "projects" / project_id
        project_paths = {"base": project_base}
        
        # Create base directory
        project_base.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in self.PROJECT_SUBDIRS:
            path = project_base / subdir
            path.mkdir(exist_ok=True)
            project_paths[subdir] = path
        
        return project_paths
    
    @handle_errors(default_return=False, error_prefix="Directory cleanup")
    def cleanup_directory(self, path: Union[str, Path], confirm: bool = True) -> bool:
        """
        Safely remove a directory and its contents.
        
        Args:
            path: Path to directory
            confirm: Whether to require confirmation
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(path)
        
        if not path.exists():
            return True
        
        if confirm and not path.is_relative_to(self.base_dir):
            raise ValueError(f"Cannot delete directory outside base: {path}")
        
        shutil.rmtree(path)
        return True
    
    def get_project_path(self, project_id: str, subdir: Optional[str] = None) -> Path:
        """
        Get path for a project or its subdirectory.
        
        Args:
            project_id: Project identifier
            subdir: Optional subdirectory name
            
        Returns:
            Path object
        """
        validate_input(project_id, str, "project_id", min_length=1)
        
        project_path = self.base_dir / "projects" / project_id
        
        if subdir:
            validate_input(subdir, str, "subdir", choices=self.PROJECT_SUBDIRS)
            return project_path / subdir
        
        return project_path
    
    @handle_errors(default_return=None, error_prefix="Temporary directory creation")
    def create_temp_directory(self, prefix: str = "openclip_") -> Optional[Path]:
        """
        Create a temporary directory.
        
        Args:
            prefix: Prefix for the temporary directory name
            
        Returns:
            Path to temporary directory or None on error
        """
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=self.base_dir / "temp"))
        return temp_dir
    
    def validate_path(
        self,
        path: Union[str, Path],
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None
    ) -> Path:
        """
        Validate a path with various constraints.
        
        Args:
            path: Path to validate
            must_exist: Whether the path must exist
            must_be_file: Whether the path must be a file
            must_be_dir: Whether the path must be a directory
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If validation fails
        """
        path = Path(path)
        
        if must_exist and not path.exists():
            raise ValueError(f"Path does not exist: {path}")
        
        if must_be_file and not path.is_file():
            raise ValueError(f"Path is not a file: {path}")
        
        if must_be_dir and not path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")
        
        if allowed_extensions and path.suffix.lower() not in allowed_extensions:
            raise ValueError(
                f"File extension {path.suffix} not allowed. "
                f"Allowed: {allowed_extensions}"
            )
        
        return path
    
    def safe_join(self, *parts: Union[str, Path]) -> Path:
        """
        Safely join path components.
        
        Args:
            *parts: Path components to join
            
        Returns:
            Joined Path object
        """
        # Filter out empty parts
        parts = [p for p in parts if p]
        
        if not parts:
            raise ValueError("No path components provided")
        
        # Convert all to Path objects and join
        result = Path(parts[0])
        for part in parts[1:]:
            result = result / part
        
        return result
    
    def get_relative_path(self, path: Union[str, Path], base: Optional[Union[str, Path]] = None) -> Path:
        """
        Get relative path from base.
        
        Args:
            path: Path to make relative
            base: Base path (defaults to self.base_dir)
            
        Returns:
            Relative Path object
        """
        path = Path(path)
        base = Path(base) if base else self.base_dir
        
        try:
            return path.relative_to(base)
        except ValueError:
            # Path is not relative to base
            return path
    
    def ensure_parent_dir(self, path: Union[str, Path]) -> Path:
        """
        Ensure parent directory exists for a file path.
        
        Args:
            path: File path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    @handle_errors(default_return=0, error_prefix="Directory size calculation")
    def get_directory_size(self, path: Union[str, Path]) -> int:
        """
        Get total size of a directory in bytes.
        
        Args:
            path: Directory path
            
        Returns:
            Size in bytes
        """
        path = Path(path)
        
        if not path.is_dir():
            return 0
        
        total_size = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """
        Format size in bytes to human-readable string.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted string (e.g., "1.5 GB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} PB"
    
    def list_files(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        recursive: bool = False,
        files_only: bool = True
    ) -> List[Path]:
        """
        List files in a directory.
        
        Args:
            directory: Directory to list
            pattern: Glob pattern for filtering
            recursive: Whether to search recursively
            files_only: Whether to return only files
            
        Returns:
            List of Path objects
        """
        directory = Path(directory)
        
        if not directory.is_dir():
            return []
        
        if recursive:
            items = directory.rglob(pattern)
        else:
            items = directory.glob(pattern)
        
        if files_only:
            return [item for item in items if item.is_file()]
        else:
            return list(items)


# Singleton instance
_path_manager = None


def get_path_manager(base_dir: Optional[Union[str, Path]] = None) -> PathManager:
    """
    Get or create the singleton PathManager instance.
    
    Args:
        base_dir: Base directory (only used on first call)
        
    Returns:
        PathManager instance
    """
    global _path_manager
    
    if _path_manager is None:
        _path_manager = PathManager(base_dir)
    
    return _path_manager 