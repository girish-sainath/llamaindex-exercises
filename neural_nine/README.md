# LlamaIndex Exercises by Neural Nine

A collection of LlamaIndex exercises demonstrating RAG, agents, tools, and vector store patterns connecting to LLMs via LiteLLM, Ollama, and Anthropic direct.

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- LLM credentials in `.env`

## Setup

```bash
uv sync
```

Create a `.env` file at the project root with your LiteLLM/SAP AI Core credentials.

## Running

```bash
uv run python main.py
```

The interactive menu offers three modes:

| Option    | Description                                                                          |
|-----------|--------------------------------------------------------------------------------------|
| `color`   | ReAct agent — answers questions about a favorite color via a function tool           |
| `rag`     | ReAct agent — answers questions over `data/` using a FAISS-backed vector store       |
| `storage` | Persistent RAG — loads a previously persisted index from the `storage/` directory    |

Individual modules can also be run directly:

```bash
uv run python src/agents/color/agent.py
uv run python src/agents/rag/agent.py
uv run python src/tools/color/tool.py
uv run python src/tools/rag_faiss/tool.py
```

The notebook provides an interactive walkthrough:

```bash
uv run jupyter lab Main.ipynb
```

## Project Structure

```text
├── main.py                          # Interactive CLI entry point
├── Main.ipynb                       # Jupyter notebook: RAG queries + FunctionAgent demo
├── data/                            # Plain-text documents used as the RAG knowledge base
├── storage/                         # Persisted LlamaIndex storage context (docstore, vector store)
└── src/
    ├── agents/
    │   ├── color/                   # ReAct agent backed by the get_favorite_color tool
    │   └── rag/                     # ReAct agent backed by the FAISS RAG tool
    ├── tools/
    │   ├── color/                   # FunctionTool: returns a hardcoded favorite color
    │   ├── rag_default/             # QueryEngineTool: in-memory VectorStoreIndex over data/
    │   ├── rag_faiss/               # QueryEngineTool: FAISS VectorStoreIndex over data/
    │   └── storage/                 # QueryEngineTool: loads a persisted StorageContext
    └── models/
        ├── ModelInfo.py             # Enum of model IDs and temperature defaults (env-overridable)
        └── ModelFactory.py          # Factory: resolves model type + access mode → LLM instance
```

## Key Concepts

- **Agents** (`src/agents/`) — each agent follows a two-file layout: `agent.py` (construction via `ReActAgent`) and an optional standalone entry point. Agents use `Context` from `llama_index.core.workflow.context` to maintain conversational state across turns.
- **Tools** (`src/tools/`) — plain Python functions wrapped as `FunctionTool` (color) or as query engine tools (RAG variants). The three RAG tools demonstrate progressive persistence: in-memory default index → FAISS vector store → pre-persisted `StorageContext`.
- **Models** (`src/models/`) — `ModelInfo` exposes model IDs as an overridable enum; `ModelFactory` resolves a `(model_type, model_access)` pair to a concrete `LiteLLM`, `Ollama`, or `Anthropic` instance. The active backend is controlled by the `MODEL_ACCESS` env var.
- **RAG** (`src/tools/rag_*/`) — documents are loaded with `SimpleDirectoryReader`, indexed via `VectorStoreIndex`, and exposed through `index.as_query_engine()`. The FAISS variant uses `FaissVectorStore(faiss.IndexFlatL2(3072))` for a persistent, efficient ANN index.
- **Notebook** (`Main.ipynb`) — standalone walkthrough: builds an index, runs queries, then creates a `FunctionAgent` with a custom tool, all using LiteLLM as the LLM and embedding backend.

## LlamaIndex

```python
"""
Tool definition example for LlamaIndex with FunctionTool
"""
from llama_index.core.tools import FunctionTool

def get_favorite_color() -> str:
    return 'Cyan'

fav_color_tool = FunctionTool.from_defaults(fn=get_favorite_color)


"""
RAG query engine tool example for LlamaIndex with LiteLLM
"""
import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

Settings.llm = LiteLLM(model='sap/anthropic--claude-4.6-sonnet')
Settings.embed_model = LiteLLMEmbedding(model_name='sap/text-embedding-3-large')

documents = SimpleDirectoryReader('data').load_data()
vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(3072))
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
query_engine = index.as_query_engine()

response = query_engine.query('What is my favorite fruit?')
print(response)


"""
Loading a persisted index from storage
"""
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir='./storage')
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()


"""
ReAct agent example for LlamaIndex with a function tool
"""
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow.context import Context

agent = ReActAgent(
    llm=LiteLLM(model='sap/anthropic--claude-4.6-sonnet'),
    tools=[fav_color_tool],
    system_prompt='You are a helpful assistant.',
)

ctx = Context(agent)
response = await agent.run('What is my favorite color?', ctx=ctx)
print(response)


"""
ReAct agent with stateful multi-turn conversation
"""
agent = ReActAgent(
    llm=LiteLLM(model='sap/anthropic--claude-4.6-sonnet'),
    tools=[fav_color_tool],
    system_prompt='You are a helpful assistant.',
)

ctx = Context(agent)
while True:
    question = input('Question: ')
    if question == 'exit':
        break
    response = await agent.run(question, ctx=ctx)
    print(response)
```
