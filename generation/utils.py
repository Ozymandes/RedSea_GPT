"""
Utility functions for RedSea GPT generation module.
"""


def clean_source_path(source: str) -> str:
    """
    Extract filename from full source path.

    Args:
        source: Full source path (e.g., "data/docs/file.pdf" or "data\\docs\\file.pdf")

    Returns:
        Clean filename (e.g., "file.pdf")

    Examples:
        >>> clean_source_path("data/docs/red_sea.pdf")
        'red_sea.pdf'
        >>> clean_source_path("data\\\\docs\\\\red_sea.pdf")
        'red_sea.pdf'
    """
    if "/" in source:
        return source.split("/")[-1]
    if "\\" in source:
        return source.split("\\")[-1]
    return source
