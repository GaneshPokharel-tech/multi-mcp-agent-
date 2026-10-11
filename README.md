# Multi MCP Agent

A local multi-agent AI application built with **LangGraph**, **Model Context Protocol (MCP)**, **Gemini**, **Streamlit**, and a custom local **PDF RAG** pipeline.

The application uses:

- one unified MCP server
- one Supervisor Agent
- four specialist agents
- specialist-specific tool boundaries
- human approval for protected actions
- persistent conversation memory
- local hybrid PDF RAG
- live execution telemetry in the UI

---

## Architecture

```text
                         User
                           │
                           ▼
                    Streamlit UI
                       app.py
                           │
                           ▼
                    AgentRuntime
                      agent.py
                           │
                           ▼
                      Supervisor
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 Productivity Agent   Developer Agent   Document Agent
        │                  │                  │
 Gmail + Calendar          Git              PDF RAG
        │
        └────────────────────────────┐
                                     │
                                     ▼
                               Utility Agent
                                     │
                           Weather / Fetch / YouTube
```

Each specialist receives only the MCP tools required for its domain.

---

# Multi-Agent Design

## Supervisor Agent

The Supervisor analyzes every request and routes it to the correct specialist.

Possible destinations:

```text
productivity
developer
document
utility
unsupported
```

The Supervisor does not directly execute MCP tools.

---

## Productivity Agent

Handles:

```text
Gmail
Google Calendar
```

### Gmail Tools

```text
search_emails
get_email
create_draft
send_email
```

### Calendar Tools

```text
list_calendar_events
create_calendar_event
update_calendar_event
delete_calendar_event
```

---

## Developer Agent

Handles Git operations.

```text
git_status
git_log
git_diff
git_show
git_branch
git_add
git_commit
git_create_branch
git_checkout
```

Read-only operations may execute directly.

Git mutations require human approval.

---

## Document Agent

Handles uploaded PDFs and document RAG.

```text
index_document
search_documents
list_documents
delete_document
```

It is responsible for:

- PDF indexing
- thread-specific document lookup
- semantic retrieval
- lexical retrieval
- page citations
- grounded PDF answers

---

## Utility Agent

Handles:

```text
Weather
Public URL Fetch
YouTube
```

Tools:

```text
get_current_weather
get_weather_forecast
fetch_url
play_youtube_song
```

---

# Unified MCP Server

The application uses one MCP server:

```text
src/multi_mcp_agent/mcp_server.py
```

Current MCP tool count:

```text
Weather             2
Gmail               4
Google Calendar     4
Git                 9
PDF RAG             4
Fetch               1
YouTube             1
----------------------
Total              25
```

No separate MCP server is created for each integration.

---

# Human Approval

Actions that modify local or external state require explicit user approval.

Protected tools:

```text
send_email

git_add
git_commit
git_create_branch
git_checkout

create_calendar_event
update_calendar_event
delete_calendar_event

delete_document
```

Example:

```text
User
 ↓
Supervisor
 ↓
Productivity Agent
 ↓
send_email
 ↓
Human Approval
      │
      ├── Approve
      │      ↓
      │   MCP Tool
      │      ↓
      │   Gmail API
      │      ↓
      │   Email sent
      │
      └── Cancel
             ↓
        Tool rejected
             ↓
       Email not sent
```

Protected actions cannot silently execute through the chat workflow.

---

# Gmail Integration

Supported Gmail operations:

```text
search_emails
get_email
create_draft
send_email
```

## Search

The agent can search email by:

- sender
- subject
- recent messages
- Gmail-compatible queries

## Read

`get_email` retrieves the full content of a selected email.

## Draft

`create_draft` creates a Gmail draft.

Creating a draft does **not** send the email.

## Send

`send_email` sends an email through Gmail.

Sending requires human approval before the MCP tool executes.

Example flow:

```text
User:
Send an email to example@example.com

        ↓

Supervisor

        ↓

Productivity Agent

        ↓

send_email

        ↓

Human Approval

        ↓

Approve

        ↓

Gmail API

        ↓

Email sent
```

OAuth credentials are stored locally in:

```text
.gmail_token.json
```

---

# Google Calendar Integration

Supported operations:

```text
list_calendar_events
create_calendar_event
update_calendar_event
delete_calendar_event
```

Calendar write operations require human approval.

The application injects the current runtime date and timezone into the agent context.

This allows requests such as:

```text
Create an event tomorrow at 3 PM.
```

to be converted into the correct absolute datetime.

Default timezone:

```text
Asia/Kathmandu
```

Calendar datetimes use RFC3339 / ISO8601-compatible values.

---

# Weather

Supported tools:

```text
get_current_weather
get_weather_forecast
```

Weather data is retrieved through WeatherAPI.

Example:

```text
User:
What's the weather in Kathmandu?

        ↓

Supervisor

        ↓

Utility Agent

        ↓

get_current_weather

        ↓

WeatherAPI

        ↓

Grounded weather response
```

---

# Public URL Fetch

Tool:

```text
fetch_url
```

The Utility Agent can retrieve and summarize public web pages.

Example:

```text
Fetch https://example.com and summarize the page.
```

The fetched page content becomes the source of truth for the answer.

---

# YouTube

Tool:

```text
play_youtube_song
```

The agent can search for and open requested songs on YouTube.

Example:

```text
Play Shape of You on YouTube.
```

The tool opens the result in the browser.

---

# PDF RAG

The application includes a custom local PDF Retrieval-Augmented Generation pipeline.

Architecture:

```text
PDF
 ↓
Page-aware text extraction
 ↓
Chunking
 ↓
Local Embeddings
 ↓
SQLite Storage
 ↓
Dense Retrieval
 +
BM25
 ↓
RRF
 ↓
MMR
 ↓
Best Chunks
 ↓
Gemini
 ↓
Grounded Answer
```

---

# Local Embeddings

Embedding model:

```text
intfloat/multilingual-e5-small
```

Embedding dimension:

```text
384
```

Embeddings are generated locally.

Benefits:

- no embedding API quota
- local document processing
- multilingual retrieval
- better privacy
- predictable document indexing

---

# Hybrid Retrieval

The RAG engine combines:

```text
Dense Semantic Retrieval
+
BM25
+
Reciprocal Rank Fusion
+
Maximum Marginal Relevance
```

## Dense Retrieval

Finds semantically related passages.

## BM25

Finds exact keywords, names, identifiers, and terminology.

## RRF

Combines dense and BM25 rankings.

## MMR

Reduces repetitive retrieval results and improves result diversity.

---

# PDF Grounding

PDF answers must come from retrieved document evidence.

Example citation:

```text
[rag_survey.pdf, page 3]
```

Example answer:

```text
Naive RAG performs retrieval before the generation step.

[rag_survey.pdf, page 3]
```

If sufficient evidence cannot be retrieved, the system returns an insufficient-evidence response rather than inventing PDF content.

---

# Multiple PDF Support

Multiple PDFs may exist inside one conversation.

Example:

```text
Chat A
 ├── rag_survey.pdf
 └── troubleshooting.pdf

Chat B
 └── architecture.pdf
```

Documents are scoped using:

```text
thread_id
```

A PDF indexed in one conversation is not automatically available in another conversation.

---

# Explicit Document Selection

When the user explicitly names a PDF:

```text
Using only rag_survey.pdf, explain Naive RAG.
```

The Document Agent:

```text
list_documents
      ↓
resolve filename
      ↓
get document_id
      ↓
search_documents
      ↓
search only selected PDF
      ↓
grounded answer
```

This prevents evidence from different documents from being mixed accidentally.

---

# Conversation Persistence

Conversation state is persisted locally using:

```text
LangGraph
+
AsyncSqliteSaver
+
SQLite
```

Database:

```text
.langgraph_checkpoints.sqlite
```

Each conversation uses its own:

```text
thread_id
```

This supports:

- saved conversations
- browser refresh persistence
- conversation switching
- follow-up questions
- approval interruption/resume
- PDF thread isolation

---

# LangGraph Workflow

High-level graph:

```text
START
  │
  ▼
Supervisor
  │
  ├─────────────┬─────────────┬─────────────┐
  │             │             │             │
  ▼             ▼             ▼             ▼
Productivity Developer    Document       Utility
  │             │             │             │
  └─────────────┴──────┬──────┴─────────────┘
                       │
                       ▼
                 Tool requested
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          Safe Tool        Protected Tool
             │                   │
             ▼                   ▼
          ToolNode            Approval
             │               /        \
             │          Approve       Reject
             │             │             │
             │             ▼             ▼
             └────────── ToolNode    Specialist
                           │
                           ▼
                      Specialist
                           │
                           ▼
                          END
```

---

# Live Agent Activity UI

The Streamlit interface displays the execution path while a request is running.

Example:

```text
Supervisor
   ↓
Developer Agent
   ↓
git_status
   ↓
Completed
```

Another example:

```text
Supervisor
   ↓
Productivity Agent
   ↓
send_email
   ↓
Waiting for approval
```

The activity panel is UI telemetry only.

It does not control routing or tool execution.

---

# Capability Boundary

The system is intentionally tool-grounded.

Supported information sources:

```text
Current conversation
Uploaded PDF documents
Weather
Gmail
Google Calendar
Git
Public URL Fetch
YouTube
```

The application is not designed as an unrestricted general-purpose chatbot.

Unsupported requests may return:

```text
That request is outside the connected capabilities of this agent.
```

---

# Project Structure

```text
multi-mcp-agent/
│
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
│
└── src/
    └── multi_mcp_agent/
        │
        ├── __init__.py
        ├── app.py
        ├── agent.py
        ├── mcp_server.py
        ├── rag.py
        │
        ├── agents/
        │   ├── __init__.py
        │   ├── base.py
        │   ├── toolsets.py
        │   ├── supervisor.py
        │   ├── productivity.py
        │   ├── developer.py
        │   ├── document.py
        │   ├── utility.py
        │   └── registry.py
        │
        ├── core/
        │   ├── __init__.py
        │   ├── config.py
        │   ├── state.py
        │   ├── persistence.py
        │   └── model.py
        │
        └── graph/
            ├── __init__.py
            ├── approval.py
            ├── routing.py
            ├── nodes.py
            └── builder.py
```

---

# File Responsibilities

## `app.py`

Streamlit frontend.

Responsible for:

```text
Chat interface
Sidebar
Conversation history
PDF uploads
Light / dark mode
Human approval controls
Live agent activity panel
Streaming responses
```

---

## `agent.py`

Application-facing runtime facade.

Responsible for:

```text
MCP adapter lifecycle
Agent registry initialization
LangGraph initialization
Runtime public API
Graph streaming
Approval resume
UI telemetry events
```

---

## `agents/`

Contains the multi-agent layer.

```text
supervisor.py
    → request classification and routing

productivity.py
    → Gmail + Calendar

developer.py
    → Git

document.py
    → PDF RAG

utility.py
    → Weather + Fetch + YouTube

toolsets.py
    → specialist tool boundaries

registry.py
    → builds the complete agent registry

base.py
    → shared specialist-agent construction
```

---

## `core/`

Shared infrastructure.

```text
config.py
    → configuration and paths

state.py
    → MultiAgentState

persistence.py
    → AsyncSqliteSaver + thread configuration

model.py
    → model retry, timeout, transient-error handling
```

---

## `graph/`

LangGraph orchestration.

```text
builder.py
    → builds the multi-agent graph

nodes.py
    → supervisor and specialist nodes

routing.py
    → graph routing decisions

approval.py
    → protected tool detection and human approval
```

---

## `mcp_server.py`

Single unified MCP execution layer.

Contains tools for:

```text
Weather
Gmail
Google Calendar
Git
Public URL Fetch
YouTube
PDF RAG
```

---

## `rag.py`

Local PDF retrieval engine.

Responsible for:

```text
PDF extraction
Page-aware chunking
Local embeddings
SQLite document storage
Dense search
BM25
RRF
MMR
Page citations
```

---

# Model

Default model:

```text
gemini-3.5-flash-lite
```

The model layer also includes:

- request timeout handling
- transient error detection
- retry handling
- rate-limit handling

---

# Security

Secrets are not hard-coded.

Local credentials:

```text
.env
.gmail_token.json
.calendar_token.json
```

Local runtime data:

```text
.langgraph_checkpoints.sqlite
.rag/
```

These files are ignored by Git.

Protected write operations require explicit human approval.

---

# Environment Variables

Create a `.env` file in the project root.

Example:

```env
WEATHER_API_KEY=your_weatherapi_key

GEMINI_API_KEY=your_gemini_api_key

GMAIL_OAUTH_CLIENT_ID=your_google_oauth_client_id
GMAIL_OAUTH_CLIENT_SECRET=your_google_oauth_client_secret

CALENDAR_TIMEZONE=Asia/Kathmandu
```

Never commit `.env`.

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd multi-mcp-agent
```

## Install Dependencies

The project uses `uv`.

```bash
uv sync
```

Optional environment activation:

```bash
source .venv/bin/activate
```

---

# Run Application

Start Streamlit:

```bash
uv run streamlit run src/multi_mcp_agent/app.py
```

---

# MCP Inspector

Run MCP Inspector:

```bash
uv run mcp dev src/multi_mcp_agent/mcp_server.py
```

Run the MCP server directly:

```bash
uv run mcp run src/multi_mcp_agent/mcp_server.py
```

---

# Validation

Compile the source:

```bash
uv run python -m compileall -q src/multi_mcp_agent
```

Lint unused imports and undefined names:

```bash
uvx ruff check src/multi_mcp_agent --select F401,F821
```

Check Git diff formatting:

```bash
git diff --check
```

---

# Verified Smoke Tests

Current architecture has been tested for:

```text
Weather
✓ Current weather

Fetch
✓ Public URL fetch

YouTube
✓ Song opening

Git
✓ git_status
✓ git_branch
✓ git_log
✓ git_diff
✓ protected action rejection

Gmail
✓ email search
✓ full email retrieval
✓ draft creation
✓ send-email rejection
✓ approved email sending

Calendar
✓ event listing
✓ protected create rejection

PDF RAG
✓ PDF upload
✓ automatic indexing
✓ document listing
✓ semantic search
✓ page citations
✓ multi-document support

Routing
✓ Supervisor routing
✓ unsupported request handling

Persistence
✓ saved conversation
✓ browser refresh
✓ follow-up context
✓ thread persistence

UI
✓ light mode
✓ dark mode
✓ conversation history
✓ delete chat
✓ live execution activity
✓ approval UI
```

---

# Known Limitations

- Gemini API availability, latency, and quota may vary.
- The first local embedding-model load may take longer.
- Scanned or image-only PDFs do not currently have OCR.
- Fetch is intended for public URLs.
- YouTube opens the browser instead of controlling playback internally.
- RAG indexes are local to the machine.
- Conversation persistence is local to the machine.
- The project is primarily designed for learning and portfolio demonstration.
- The system intentionally does not act as an unrestricted general-purpose chatbot.

---

# Key Learning Outcomes

This project demonstrates:

```text
Model Context Protocol
Multi-agent architecture
Supervisor routing
Specialist tool isolation
LangGraph orchestration
LangChain MCP integration
Tool calling
Human-in-the-loop approval
Persistent graph state
OAuth integration
Gmail automation
Calendar automation
Git automation
Weather API integration
Local PDF RAG
Local embeddings
Hybrid retrieval
Multi-document retrieval
Source-grounded answers
Streamlit agent observability
```

---

# Execution Model

```text
User
 ↓
Streamlit UI
 ↓
AgentRuntime
 ↓
Supervisor
 ↓
Specialist Agent
 ↓
LangGraph
 ↓
MCP Tool
 ↓
External API / Local System
 ↓
Tool Result
 ↓
Specialist Agent
 ↓
Streamlit UI
 ↓
User
```

---

# Summary

`Multi MCP Agent` evolved from a basic weather MCP learning project into a modular multi-agent system.

The final architecture separates:

```text
UI
↓
Runtime
↓
Supervisor
↓
Specialist Agents
↓
LangGraph
↓
MCP
↓
External Services / Local Systems
```

This keeps the project easier to understand, safer to operate, and easier to extend with additional MCP capabilities without turning the system into one large monolithic agent.