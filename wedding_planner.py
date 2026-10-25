"""Multi-agent destination wedding planner.

A coordinator agent collects the trip details, stores them in state, then
delegates to three specialists: flights (Kiwi MCP server), venues (Tavily web
search) and music (SQL queries on the Chinook sample database).

Run:  python wedding_planner.py "I'm from London and want a wedding in Paris for 100 guests, jazz"
"""

import asyncio
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

# Windows needs the Proactor loop to launch subprocesses (MCP) from Python.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

MODEL = "gpt-5-nano"
DB_PATH = "sqlite:///resources/Chinook.db"
RETRYABLE_MCP_CODES = {-32603}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str


# ---------------------------------------------------------------------------
# Tools used by the specialists
# ---------------------------------------------------------------------------
class RetryMCPInterceptor:
    """Retry flaky MCP calls and return errors as text so the agent can adapt."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed: {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2**attempt)
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


tavily_client = TavilyClient()


@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. Track your search count by passing
    search_number (starting at 1) and max_search_number on every call.
    Use plain ASCII characters in queries."""
    if search_number > max_search_number:
        return {"message": "Search limit reached. Summarize your findings and give your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}


db = SQLDatabase.from_uri(DB_PATH)


@tool
def query_playlist_db(query: str) -> str:
    """Query the SQLite music database for playlist information."""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
async def build_coordinator():
    client = MultiServerMCPClient(
        {"travel_server": {"transport": "streamable_http", "url": "https://mcp.kiwi.com"}},
        tool_interceptors=[RetryMCPInterceptor()],
    )
    flight_tools = await client.get_tools()

    travel_agent = create_agent(
        model=MODEL,
        tools=flight_tools,
        system_prompt="""
        You are a travel agent. Search for flights to the wedding destination.
        Do not ask follow up questions. Pick the best options by:
        - Price (lowest, economy)
        - Duration (shortest)
        - Date (best time of year for a wedding at this location)
        Only look for one ticket, one way. Make several searches if needed.
        If the tool fails or returns unusable output, try again.
        Finish with a short list of your best options.
        """,
    )

    venue_agent = create_agent(
        model=MODEL,
        tools=[web_search],
        system_prompt="""
        You are a venue specialist. Find venues in the destination that fit the guest count.
        Do not ask follow up questions. Pick the best options by:
        - Price (lowest)
        - Capacity (exact match)
        - Reviews (highest)
        You may make up to 12 web searches. Count every call.
        After 12 searches, stop and summarize the best options found.
        """,
    )

    playlist_agent = create_agent(
        model=MODEL,
        tools=[query_playlist_db],
        system_prompt="""
        You are a playlist specialist. Query the SQL database and curate a wedding
        playlist for the given genre. Then compute the total duration and total cost
        (each song has a price). If a query fails, fix it and retry.
        Do not come back empty handed.
        This is a SQLite database. Discover the schema before writing data queries.
        """,
    )

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the wedding destination."""
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        result = await travel_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]}
        )
        return result["messages"][-1].content

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent picks the best venues for the destination and guest count."""
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]
        result = venue_agent.invoke(
            {"messages": [HumanMessage(content=f"Find wedding venues in {destination} for {capacity} guests")]}
        )
        return result["messages"][-1].content

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates a playlist for the genre."""
        genre = runtime.state["genre"]
        result = playlist_agent.invoke(
            {"messages": [HumanMessage(content=f"Find {genre} tracks for a wedding playlist")]}
        )
        return result["messages"][-1].content

    @tool
    def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> Command:
        """Save origin, destination, guest_count and genre once all four are known.
        Call this tool alone, and wait for it to return before using other tools."""
        return Command(
            update={
                "origin": origin,
                "destination": destination,
                "guest_count": guest_count,
                "genre": genre,
                "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)],
            }
        )

    return create_agent(
        model=MODEL,
        tools=[search_flights, search_venues, suggest_playlist, update_state],
        state_schema=WeddingState,
        system_prompt="""
        You are a wedding coordinator.
        First gather everything needed to update the state, then update it.
        Once that has returned, delegate to your specialists for flights, venues and playlists.
        When you have their answers, put together the perfect wedding plan.
        """,
    )


async def main(request: str) -> None:
    coordinator = await build_coordinator()
    response = await coordinator.ainvoke(
        {"messages": [HumanMessage(content=request)]},
        config={"tags": ["WP"], "recursion_limit": 40},
    )
    print(response["messages"][-1].content)


if __name__ == "__main__":
    default = "I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre"
    asyncio.run(main(" ".join(sys.argv[1:]) or default))
