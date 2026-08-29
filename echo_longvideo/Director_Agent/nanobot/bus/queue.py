"""Async message queue for decoupled channel-agent communication."""

import asyncio

from nanobot.bus.events import InboundMessage, OutboundMessage, RuntimeEvent


class MessageBus:
    """
    Async message bus that decouples chat channels from the agent core.

    Channels push messages to the inbound queue, and the agent processes
    them and pushes responses to the outbound queue.
    """

    def __init__(self):
        self.inbound: asyncio.Queue[InboundMessage] = asyncio.Queue()
        self.outbound: asyncio.Queue[OutboundMessage] = asyncio.Queue()
        self.runtime: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=1000)

    async def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> InboundMessage:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: OutboundMessage) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> OutboundMessage:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    async def publish_runtime(self, event: RuntimeEvent) -> None:
        """Publish an internal runtime event that is not consumed by the agent loop."""
        if self.runtime.full():
            try:
                self.runtime.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self.runtime.put(event)

    async def consume_runtime(self) -> RuntimeEvent:
        """Consume the next runtime event (blocks until available)."""
        return await self.runtime.get()

    @property
    def inbound_size(self) -> int:
        """Number of pending inbound messages."""
        return self.inbound.qsize()

    @property
    def outbound_size(self) -> int:
        """Number of pending outbound messages."""
        return self.outbound.qsize()

    @property
    def runtime_size(self) -> int:
        """Number of pending internal runtime events."""
        return self.runtime.qsize()
