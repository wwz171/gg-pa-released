"""RPC transport stub (not implemented)."""
from __future__ import annotations

from typing import Dict, Iterable

from ggpa.core.protocol import ClientReply, ClientRequest
from ggpa.core.interfaces import Transport
from ggpa.core.errors import NotSupportedError


class RPCTransport(Transport):
    """Placeholder for future RPC transport."""

    def call(self, client_id: str, request: ClientRequest) -> ClientReply:
        raise NotSupportedError("RPCTransport is not implemented")

    def call_all(self, requests: Iterable[ClientRequest]) -> Dict[str, ClientReply]:
        """Call all clients and return dict.
        
        Returns:
            Dict mapping client_id to ClientReply
        """
        replies = {}
        for req in requests:
            replies[req.client_id] = self.call(req.client_id, req)
        return replies
