import logging
import os
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Enum representing supported file types for document processing."""
    PDF = auto()
    DOCX = auto()
    DOC = auto()
    EXCEL = auto()
    CSV = auto()
    ZIP = auto()
    HTML = auto()
    TEXT = auto()
    UNSUPPORTED = auto()


def get_file_type(file_path_or_ext: str) -> FileType:
    """
    Determine file type from file path or extension.
    
    Args:
        file_path_or_ext: Either a file path or just a file extension
        
    Returns:
        FileType enum indicating the type of file
    """
    # Extract extension if given a file path
    if os.path.isfile(file_path_or_ext) or '/' in file_path_or_ext or '\\' in file_path_or_ext:
        ext = os.path.splitext(file_path_or_ext)[1].lower()
    else:
        # Assume it's just an extension
        ext = file_path_or_ext.lower() if file_path_or_ext.startswith('.') else f".{file_path_or_ext.lower()}"

    # Map extensions to file types
    if ext == ".pdf":
        return FileType.PDF
    elif ext == ".docx":
        return FileType.DOCX
    elif ext == ".doc":
        return FileType.DOC
    elif ext in [".xls", ".xlsx"]:
        return FileType.EXCEL
    elif ext == ".csv":
        return FileType.CSV
    elif ext == ".zip":
        return FileType.ZIP
    elif ext in [".txt", ".text"]:
        return FileType.TEXT
    elif ext in [".html", ".htm"]:
        return FileType.HTML
    else:
        logger.warning(f"Unsupported file extension: {ext}")
        return FileType.UNSUPPORTED


def get_file_extension(file_type: FileType) -> str:
    """
    Get default file extension for a given file type.
    
    Args:
        file_type: FileType enum value
        
    Returns:
        Default extension string for the file type (including the dot)
    """
    extension_map = {
        FileType.PDF: ".pdf",
        FileType.DOCX: ".docx",
        FileType.DOC: ".doc",
        FileType.EXCEL: ".xlsx",
        FileType.CSV: ".csv",
        FileType.ZIP: ".zip",
        FileType.HTML: ".html",
        FileType.TEXT: ".txt",
        FileType.UNSUPPORTED: ".bin"
    }
    return extension_map.get(file_type, ".bin")


def guess_file_type_from_content_type(content_type: str) -> Optional[FileType]:
    """
    Attempt to determine file type from HTTP Content-Type header.
    
    Args:
        content_type: HTTP Content-Type header value
        
    Returns:
        FileType enum or None if cannot be determined
    """
    content_type = content_type.lower()

    if "pdf" in content_type:
        return FileType.PDF
    elif "wordprocessingml" in content_type or "docx" in content_type:
        return FileType.DOCX
    elif "msword" in content_type:
        return FileType.DOC
    elif "spreadsheetml" in content_type or "xlsx" in content_type:
        return FileType.EXCEL
    elif "excel" in content_type or "xls" in content_type:
        return FileType.EXCEL
    elif "csv" in content_type:
        return FileType.CSV
    elif "zip" in content_type:
        return FileType.ZIP
    elif "html" in content_type:
        return FileType.HTML
    elif "text/plain" in content_type:
        return FileType.TEXT

    return None
