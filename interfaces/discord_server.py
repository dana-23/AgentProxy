import logging

import discord
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class DiscordServer:
    def __init__(self, graph: CompiledStateGraph, channel_name: str = "agent-chat", debug: bool = False):
        self.graph = graph
        self.channel_name = channel_name

        logging.basicConfig(
            level=logging.DEBUG if debug else logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)

        self._register_events()

    def _register_events(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Bot is online as {self.client.user}")

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            mentioned = self.client.user in message.mentions
            in_agent_channel = message.channel.name == self.channel_name
            if not (mentioned or in_agent_channel):
                return

            query = message.content.replace(f"<@{self.client.user.id}>", "").strip()
            if not query:
                return

            config = {"configurable": {"thread_id": str(message.author.id)}}
            logger.debug(f"[{message.author}] {query}")

            async with message.channel.typing():
                result = await self.graph.ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config=config,
                )
                logger.debug(f"Graph returned {len(result['messages'])} messages")

                content = result["messages"][-1].content
                if isinstance(content, list):
                    reply = next(
                        (block["text"] for block in content if block.get("type") == "text"),
                        "",
                    )
                else:
                    reply = content

            if len(reply) <= 2000:
                await message.reply(reply)
            else:
                for i in range(0, len(reply), 2000):
                    await message.channel.send(reply[i : i + 2000])

    def run(self, token: str):
        self.client.run(token)
