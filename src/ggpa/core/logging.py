"""Logging utilities for GG-PA.

This module provides centralized logging configuration for the GG-PA framework.

Logging hierarchy:
    ggpa                  # Root logger
    ├── ggpa.kernel       # Kernel operations
    ├── ggpa.server       # Server operations  
    ├── ggpa.client       # Client operations
    ├── ggpa.transport    # Transport/communication
    └── ggpa.aggregation  # Aggregation strategies

Levels:
    - DEBUG: Detailed internal state, gradients, samples
    - INFO: Operation summaries, step progress, method used
    - WARNING: Unusual conditions, convergence issues
    - ERROR: Failures, exceptions

Usage:
    from ggpa.core.logging import get_logger
    
    logger = get_logger("kernel")  # Creates "ggpa.kernel" logger
    logger.info("Step 0: tau=0.5, ||s||=3.14")
    logger.debug(f"State details: s.shape={s.shape}, s.mean={s.mean()}")
"""
from __future__ import annotations

import logging
from typing import Optional


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the 'ggpa' prefix.
    
    Args:
        name: Component name (e.g., 'kernel', 'server', 'client')
              Creates logger 'ggpa.<name>'
    
    Returns:
        Logger instance with 'ggpa.<name>' namespace
    
    Example:
        >>> logger = get_logger("kernel")
        >>> logger.name
        'ggpa.kernel'
    """
    return logging.getLogger(f"ggpa.{name}")


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    date_format: Optional[str] = None
) -> None:
    """Setup global logging configuration for GG-PA.
    
    This configures the root 'ggpa' logger and all child loggers.
    Call this once at the start of your program.
    
    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
               Default: logging.INFO
        format_string: Custom format string
                       Default: "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        date_format: Custom date format
                     Default: "%Y-%m-%d %H:%M:%S"
    
    Example:
        # Basic setup (INFO level)
        setup_logging()
        
        # Debug mode
        setup_logging(level=logging.DEBUG)
        
        # Custom format
        setup_logging(
            level=logging.DEBUG,
            format_string="%(levelname)s | %(name)s | %(message)s"
        )
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configure root ggpa logger
    logger = logging.getLogger("ggpa")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Add console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Set formatter
    formatter = logging.Formatter(fmt=format_string, datefmt=date_format)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False

