"""Serializable request/response structures for Request/Reply pattern.

ARCHITECTURE: Unified Request/Reply communication with:
- Batch requests (request_types: Union[str, List[str]])
- Status codes ('success', 'error', 'partial', 'unsupported')  
- Flexible data dictionary (keyed by request_type)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ClientRequest:
    """Request from Server to Client (unified Request/Reply pattern).
    
    Supports batch requests for multiple types of data in one call.
    
    Attributes:
        client_id: Unique client identifier
        s: Signal in signal space (client handles projection)
        tau: Diffusion time in [0, 1]
        request_types: Single or list of request types:
            - 'sample': Get denoised sample x ~ p(x | y, tau)
            - 'gradient': Get gradient ∇_s log p(Φ(s))
            - 'log_prob': Get log probability log p(Φ(s) | tau)
            - 'properties': Query client metadata
        request_id: Optional ID for tracking
        seed: Optional random seed for reproducibility
        step: Optional step number for diagnostics
        metadata: Optional extra data
    """
    client_id: str
    s: Any
    tau: float
    request_types: Union[str, List[str]]
    request_id: Optional[str] = None
    seed: Optional[int] = None
    step: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize request_types to list for uniform handling."""
        if isinstance(self.request_types, str):
            self.request_types = [self.request_types]


@dataclass
class ClientReply:
    """Reply from Client to Server (unified Request/Reply pattern).
    
    Returns requested data with status code for error handling.
    
    Attributes:
        client_id: Unique client identifier
        request_id: Optional ID matching the request
        status_code: Result status:
            - 'success': All requested data successfully provided
            - 'error': Exception occurred (see error field)
            - 'partial': Some requests succeeded, others returned None
            - 'unsupported': All request types unsupported
        data: Dictionary mapping request_type to result:
            - 'sample' -> x (denoised sample)
            - 'gradient' -> grad (gradient in signal space)
            - 'log_prob' -> float (log probability)
            - 'properties' -> dict (metadata)
            - Unsupported types -> None
        error: Error message if status_code != 'success'
        diagnostics: Optional diagnostic information
    """
    client_id: str
    request_id: Optional[str]
    status_code: str  # 'success', 'error', 'partial', 'unsupported'
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if reply is successful (all data provided)."""
        return self.status_code == 'success'
