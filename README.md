# Destination Wedding Planner

A multi-agent system built with LangChain. A coordinator agent collects the
trip details (origin, destination, guest count, music genre), saves them in
state, then delegates to three specialist agents:

- Travel agent: finds flights through the Kiwi MCP server
- Venue agent: searches the web for venues with Tavily
- Playlist agent: queries the Chinook SQLite database for a playlist

Based on the project in Module 2 of the LangChain Academy course
"Introduction to LangChain (Python)".

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
cp example.env .env           # then fill in your keys
```

Create a folder named `resources` and put `Chinook.db` inside it. The file is
in the course repository under `notebooks/module-2/resources/`.

## Run

```bash
python wedding_planner.py "I'm from Toronto and want a wedding in Rome for 80 guests, jazz"
```

A run can take a few minutes because three sub agents work in turn. Turn on
LangSmith tracing to see what each one did.

## Notes

Never commit `.env`. It holds your API keys and is listed in `.gitignore`.
