# Agents SDK Basics

An agent combines instructions, a model, and optional tools or handoffs. The
OpenAI Agents SDK's `Runner` executes the agent and coordinates model responses,
tool calls, and transitions to other agents.

A function tool lets an agent ask the application to run a defined function.
The application executes that function and returns its result to the agent.
Tools are useful when the model needs application data or must request a
controlled operation.

A handoff transfers responsibility for the current request to another agent.
The receiving agent is a specialist with its own instructions and tools. A
handoff is different from a function tool: it changes which agent handles the
conversation, while a tool asks the application to perform a function.

An input guardrail checks a user's input before the main agent proceeds. An
output guardrail checks the agent's response. Tool guardrails can validate
arguments before a tool runs or inspect its result afterward. Guardrails should
be combined with deterministic application-side authorization and validation.

Structured outputs constrain a response to a schema, often represented by a
Pydantic model. The application can then validate and work with typed fields
instead of parsing arbitrary prose. Provider APIs may differ in which schema
features they accept.

Conversation sessions can preserve prior turns between runs. Application
context is separate: it carries trusted runtime data, such as a user account
object, to tools. Do not treat model-generated text or conversation history as
trusted authorization data.
