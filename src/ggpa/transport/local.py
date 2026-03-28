"""Transport implementations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from ggpa.core.interfaces import Client
from ggpa.core.logging import get_logger
from ggpa.core.protocol import ClientReply, ClientRequest
from ggpa.core.interfaces import Transport

logger = get_logger("transport")


@dataclass
class LocalTransport(Transport):
    """In-process transport using direct method calls.
    
    Uses the unified Request/Reply pattern via handle_request().
    """

    clients: Dict[str, Client]

    def call(self, client_id: str, request: ClientRequest) -> ClientReply:
        """Call a single client using Request/Reply pattern."""
        logger.debug(f"Calling client {client_id} with request_types={request.request_types}")
        client = self.clients[client_id]
        reply = client.handle_request(request)
        logger.debug(f"Client {client_id} replied with status={reply.status_code}")
        return reply

    def call_all(self, requests: Iterable[ClientRequest]) -> Dict[str, ClientReply]:
        """Call all clients sequentially.
        
        Returns:
            Dict mapping client_id to ClientReply
        """
        requests_list = list(requests)
        logger.info(f"Broadcasting to {len(requests_list)} clients")
        replies = {req.client_id: self.call(req.client_id, req) for req in requests_list}
        
        # Summary statistics
        success_count = sum(1 for r in replies.values() if r.status_code == 'success')
        logger.info(f"Broadcast complete: {success_count}/{len(replies)} successful")
        
        return replies
