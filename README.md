# OpenAI Agents SDK Learning Project

A collection of small Python examples for learning agent workflows with the
[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/). The examples
use Groq's OpenAI-compatible API as the model provider, so they do not send model
requests to the OpenAI API.

## What this project demonstrates

- Creating an agent and running it asynchronously with `Runner`
- Registering Python functions as agent tools
- Routing requests to specialist agents with handoffs
- Streaming responses and run events
- Returning validated, typed data with Pydantic structured outputs
- Passing application state to tools and retaining conversation history
- Input, output, and tool guardrails
- Coordinating writer, reviewer, and publisher agents in an evaluator-optimizer
  workflow
- A first RAG workflow: searching local Markdown documents, grounding answers in
  retrieved passages, and citing their source chunks

## Project layout

| File | Purpose |
| --- | --- |
| [`agent.py`](./agent.py) | Minimal history tutor agent |
| [`tools.py`](./tools.py) | Function tools and tool registration |
| [`handoff.py`](./handoff.py) | Triage agent routing to history or math specialists |
| [`repl_demo.py`](./repl_demo.py) | Interactive multi-agent chat loop |
| [`streaming.py`](./streaming.py) | Streaming tokens, tool events, and agent handoffs |
| [`structured_outputs.py`](./structured_outputs.py) | Extracting an order into Pydantic models |
| [`context_state.py`](./context_state.py) | Runtime application context, state-changing tools, and SQLite conversation session |
| [`rag.py`](./rag.py) | Local document retrieval, evidence-grounded answers, and citations |
| [`knowledge_base/agents_sdk_basics.md`](./knowledge_base/agents_sdk_basics.md) | Sample document searched by the RAG example |
| [`tests/test_rag.py`](./tests/test_rag.py) | Offline unit tests for document loading and retrieval |
| [`Guardrails/Input_guard.py`](./Guardrails/Input_guard.py) | Input guardrail example |
| [`Guardrails/Output_guard.py`](./Guardrails/Output_guard.py) | Output guardrail example |
| [`Guardrails/tools_guard.py`](./Guardrails/tools_guard.py) | Tool input and output guardrails |
| [`evaluator_optimizer_graph.py`](./evaluator_optimizer_graph.py) | Streaming writer-reviewer-publisher workflow |
| [`evaluator_optimizer_graph.ipynb`](./evaluator_optimizer_graph.ipynb) | Notebook version of the evaluator-optimizer example |

## Requirements

- Python 3.10 or newer
- A [Groq API key](https://console.groq.com/keys)
- A Groq account with access to the model configured in the examples
  (`qwen/qwen3.8-27b` at the time of writing)

The examples use the Agents SDK, the OpenAI Python client, Pydantic, and
`python-dotenv`.

## Setup (Windows PowerShell)

From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install openai-agents python-dotenv pydantic
```

Create a `.env` file in the project root and add your key:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep `.env` private; do not commit API keys. The examples load it with
`python-dotenv`. They configure the OpenAI client to use Groq's API endpoint,
disable Agents SDK tracing, and remove `OPENAI_API_KEY` to avoid accidental
OpenAI tracing requests.

## Run an example

Activate the virtual environment, then run a script from the project root:

```powershell
python .\agent.py
python .\tools.py
python .\handoff.py
python .\streaming.py
python .\structured_outputs.py
python .\context_state.py
python .\rag.py
python .\Guardrails\Input_guard.py
python .\Guardrails\Output_guard.py
python .\Guardrails\tools_guard.py
python .\evaluator_optimizer_graph.py
```

Run `repl_demo.py` for the interactive chat loop:

```powershell
python .\repl_demo.py
```

Each script is an independent example. You do not need to run them in sequence.
The weather function in `streaming.py` returns sample data; it does not call a
live weather service.

The RAG example uses a simple keyword-overlap search over paragraphs in
`knowledge_base\`. It requires no vector database or embedding model. It is
intended to demonstrate the retrieval and grounding flow, not production-grade
search relevance.

## Notes and limitations

- These are teaching examples, not a production-ready application.
- Model names and provider support can change. If Groq reports that a model is
  unavailable, check the models enabled for your account and update the model
  configuration in the relevant script.
- OpenAI-compatible providers do not necessarily support every OpenAI API
  feature or schema identically. Structured outputs, tool calls, and handoff
  schemas can fail validation when the provider's supported JSON Schema differs
  from what the SDK sends.
- The guardrails demonstrate SDK wiring, not a complete safety system. In
  particular, keyword matching is not reliable secret detection, and an LLM
  classifier is not a substitute for deterministic validation or authorization.
- The account and purchase data in `context_state.py` are hard-coded demo data.
  Do not use this pattern to authorize real financial transactions.

## What to learn next

Suggested learning path:

1. **Agent fundamentals:** instructions, model settings, run lifecycle, run
   results, and how the SDK handles tool calls.
2. **Tools and safe side effects:** input validation, typed arguments, clear
   error handling, idempotency, timeouts, and authorization checks before
   changing external state.
3. **Handoffs and orchestration:** routing criteria, shared context, loop limits,
   failure paths, and when a single agent is preferable to multiple agents.
4. **Sessions and memory:** persistent sessions, what should be stored, data
   retention, and separating conversation history from trusted application
   state.
5. **Structured outputs:** Pydantic validation, optional fields, schema
   constraints, provider compatibility, and handling invalid or incomplete
   responses.
6. **Guardrails and security:** input/output validation, prompt-injection
   risks, secret handling, least privilege, and deterministic policy enforcement.
7. **Testing and evaluation:** unit-test tools and policy logic, mock model
   calls, create representative evaluation datasets, and measure quality,
   latency, and cost.
8. **Observability and production readiness:** tracing, structured logs,
   retries, rate limits, concurrency, human approval for consequential actions,
   and safe deployment of API keys.
9. **RAG beyond the baseline in `rag.py`:** improve chunking, compare keyword
   search with embeddings and vector databases, add metadata filters, and
   evaluate retrieval quality and citation faithfulness.

## References

- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK GitHub repository](https://github.com/openai/openai-agents-python)
- [Groq documentation](https://console.groq.com/docs/overview)
