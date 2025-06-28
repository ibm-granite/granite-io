# API Reference

This page documents the main public API of the `granite-io` framework. It is organized by module and includes class/function signatures, docstrings, and usage examples.

---

## Top-level API

```python
from granite_io import (
    make_backend, make_io_processor,
    get_input_processor, get_output_processor,
    input_processor, io_processor, output_processor,
    ModelDirectInputOutputProcessor, make_new_io_processor,
    ChatCompletionInputs, ChatCompletionResults, GenerateResult, GenerateResults, UserMessage
)
```

### Example Usage

```python
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
)
messages = [UserMessage(content="What's the fastest way for a seller to visit all the cities in their region?")]
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print(outputs.results[0].next_message.content)
```

---

## Types

The `granite_io.types` module provides comprehensive data models for working with chat completions, model inputs/outputs, and structured data like citations and hallucinations.

### Core Data Models

#### Chat Messages

**UserMessage**
- Represents a user's message in a conversation
- Fields: `content` (str), `role` (always "user")

**AssistantMessage**
- Represents the model's response
- Fields:
  - `content` (str): The main response text
  - `role` (always "assistant")
  - `tool_calls` (list[FunctionCall]): Function calls made by the model
  - `reasoning_content` (str | None): Model's reasoning/thinking process
  - `citations` (list[Citation] | None): References to source documents
  - `documents` (list[Document] | None): Source documents used
  - `hallucinations` (list[Hallucination] | None): Detected hallucinations
  - `stop_reason` (str | None): Why the model stopped generating
  - `raw` (property): Raw response content before parsing

**SystemMessage**
- Represents system instructions
- Fields: `content` (str), `role` (always "system")

**ToolResultMessage**
- Represents results from function/tool calls
- Fields: `content` (str), `role` (always "tool"), `tool_call_id` (str)

#### Function Calling

**FunctionCall**
- Represents a function call made by the model
- Fields:
  - `id` (str | None): Unique identifier for the call
  - `name` (str): Name of the function to call
  - `arguments` (dict[str, Any] | None): Arguments for the function

**FunctionDefinition**
- Defines a function that the model can call
- Fields:
  - `name` (str): Function name
  - `description` (str | None): Function description
  - `parameters` (dict[str, Any] | None): JSON schema for parameters
- Methods:
  - `to_openai_json()`: Convert to OpenAI-compatible format

#### RAG and Citations

**Document**
- Represents a source document for RAG
- Fields:
  - `doc_id` (str): Unique document identifier
  - `text` (str): Document content

**Citation**
- Represents a citation in the model's response
- Fields:
  - `citation_id` (str): Unique citation identifier
  - `doc_id` (str): Reference to source document
  - `context_text` (str): Text from the source document
  - `context_begin` (int): Start position in source document
  - `context_end` (int): End position in source document
  - `response_text` (str): Text in the response being cited
  - `response_begin` (int): Start position in response
  - `response_end` (int): End position in response

#### Hallucination Detection

**Hallucination**
- Represents detected hallucination in model output
- Fields:
  - `hallucination_id` (str): Unique hallucination identifier
  - `risk` (str): Risk level of the hallucination
  - `reasoning` (str | None): Explanation of why it's flagged
  - `response_text` (str): The hallucinated text
  - `response_begin` (int): Start position in response
  - `response_end` (int): End position in response

#### Input/Output Models

**ChatCompletionInputs**
- Main input class for chat completion requests
- Fields:
  - `messages` (list[ChatMessage]): Conversation history
  - `tools` (list[FunctionDefinition]): Available functions
  - `generate_inputs` (GenerateInputs | None): Generation parameters
- Methods:
  - `with_messages(new_messages)`: Create copy with new message list
  - `with_next_message(message)`: Add message to conversation
  - `with_addl_generate_params(params)`: Add generation parameters

**ChatCompletionResults**
- Container for multiple chat completion results
- Fields: `results` (list[ChatCompletionResult])

**ChatCompletionResult**
- Single chat completion result
- Fields: `next_message` (ChatMessage)

**GenerateInputs**
- Low-level generation parameters for backends
- Fields include all standard generation parameters:
  - `prompt` (str | list): Input prompt(s)
  - `model` (str): Model name/ID
  - `max_tokens` (int): Maximum tokens to generate
  - `temperature` (float): Randomness control (0.0-1.0)
  - `top_p` (float): Nucleus sampling parameter
  - `n` (int): Number of completions to generate
  - `stop` (str | list[str]): Stop sequences
  - `frequency_penalty` (float): Penalize frequent tokens
  - `presence_penalty` (float): Penalize new tokens
  - `logit_bias` (dict): Token bias adjustments
  - `stream` (bool): Enable streaming
  - `user` (str): User identifier
  - `extra_headers` (dict): Additional HTTP headers
  - `extra_body` (dict): Additional request body data

**GenerateResults**
- Container for multiple generation results
- Fields: `results` (list[GenerateResult])

**GenerateResult**
- Single generation result from backend
- Fields:
  - `completion_string` (str): Generated text
  - `completion_tokens` (list[int]): Token IDs (if available)
  - `stop_reason` (str): Why generation stopped

### Usage Examples

```python
from granite_io.types import (
    UserMessage, AssistantMessage, ChatCompletionInputs,
    Document, Citation, FunctionDefinition
)

# Create a simple chat input
messages = [
    UserMessage(content="What is the capital of France?")
]
inputs = ChatCompletionInputs(messages=messages)

# Add documents for RAG
documents = [
    Document(doc_id="doc1", text="Paris is the capital of France...")
]

# Define a function the model can call
function = FunctionDefinition(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        }
    }
)

# Create inputs with RAG and function calling
inputs = ChatCompletionInputs(
    messages=messages,
    documents=documents,
    tools=[function]
)
```

---

## Input/Output Processors

### Processor Abstractions

- `InputOutputProcessor`: Abstract base for chat completion processors.
- `ModelDirectInputOutputProcessor`: Base for processors that translate inputs to strings and parse outputs.
- `InputProcessor`, `OutputProcessor`: Abstract base classes for input/output transformation.
- `make_new_io_processor(input_processor, output_processor, ...)`: Compose a new IO processor from input/output processors.

### Registry Decorators and Constructors

- `@io_processor`, `@input_processor`, `@output_processor`: Decorators to register new processors.
- `make_io_processor`, `get_input_processor`, `get_output_processor`: Factory functions to construct processors by name or config.

#### Example: Registering a Custom IO Processor

```python
from granite_io.io import io_processor

@io_processor("my_custom_processor")
class MyCustomProcessor(InputOutputProcessor):
    ...
```

---

## Backends

### Backend Abstractions

- `Backend`: Base class for string-based completions API.
- `ChatCompletionBackend`: Base for chat completion APIs.

### Built-in Backends

- `OpenAIBackend`, `LiteLLMBackend`, `TransformersBackend`, `LocalVLLMServer`
- Register or construct with `@backend` decorator and `make_backend` function.

#### Example: Using a Backend

```python
from granite_io import make_backend
backend = make_backend("openai", {"model_name": "granite3.2:8b"})
```

---

## Visualization

### Visualization Utilities

- `CitationsWidget`, `CitationsWidgetInstance`: Jupyter widgets for visualizing citations in RAG outputs.

#### Example

```python
from granite_io.visualization import CitationsWidget
widget = CitationsWidget()
widget.show(inputs, outputs)
```

---

## Utilities

### Optional Dependency Helpers

- `import_optional(extra_name: str)`: Context manager for optional imports.
- `nltk_check(feature_name: str)`: Context manager for NLTK-dependent features.

---

## API in Practice: Example Scripts

This section provides a comprehensive, example-driven guide to the `granite-io` API. Each script in the `/examples/` folder is described, and every operator, method, and class used is documented in detail, with code signatures, descriptions, and usage context. This complements the main API reference and demonstrates real-world usage.

### model_chat.py

**Purpose:**
Basic chat completion using a Granite model and the OpenAI backend (Ollama server).

**Key API Elements:**
- `make_backend(name: str, config: dict)`
  - Factory function to create a backend for model inference.
  - Example: `make_backend("openai", {"model_name": model_name})`
- `make_io_processor(model_name: str, backend: Backend)`
  - Factory function to create an input/output processor for a model.
  - Example: `make_io_processor(model_name, backend=...)`
- `UserMessage(content: str)`
  - Represents a user message in a chat.
- `ChatCompletionInputs(messages: list[ChatMessage])`
  - Input object for chat completion.
- `io_processor.create_chat_completion(inputs: ChatCompletionInputs)`
  - Runs the chat completion and returns results.

**Usage:**
```python
model_name = "granite3.2:8b"
io_processor = make_io_processor(model_name, backend=make_backend("openai", {"model_name": model_name}))
question = "Find the fastest way for a seller to visit all the cities in their region"
messages = [UserMessage(content=question)]
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print(outputs.results[0].next_message.content)
```

---

### model_chat_with_citation.py

**Purpose:**
Chat completion with retrieval-augmented generation (RAG) and citation extraction.

**Key API Elements:**
- All from `model_chat.py`, plus:
- `documents` parameter in `ChatCompletionInputs`
  - List of dicts with document text for RAG.
- `controls={"citations": True}`
  - Enables citation extraction in the output.
- `outputs.results[0].next_message.citations`
  - List of citations extracted from the model output.
- `outputs.results[0].next_message.documents`
  - List of documents referenced in the output.

**Usage:**
```python
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        documents=documents,
        controls={"citations": True},
    )
)
if outputs.results[0].next_message.citations:
    pprint.pprint(outputs.results[0].next_message.citations)
```

---

### model_chat_with_thinking.py

**Purpose:**
Chat completion with "thinking" mode enabled, which provides model reasoning content in addition to the response.

**Key API Elements:**
- `thinking=True` in `ChatCompletionInputs`
  - Enables model reasoning output.
- `outputs.results[0].next_message.reasoning_content`
  - The model's thought process or reasoning.

**Usage:**
```python
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print(outputs.results[0].next_message.reasoning_content)
```

---

### model_chat_with_thinking_custom_io_processor.py

**Purpose:**
Demonstrates creating a custom IO processor by subclassing `ModelDirectInputOutputProcessor` and using specific input/output processors.

**Key API Elements:**
- `ModelDirectInputOutputProcessor`
  - Base class for custom IO processors.
- `get_input_processor(model_name: str)`
  - Retrieves an input processor for a given model.
- `get_output_processor(model_name: str)`
  - Retrieves an output processor for a given model.
- Custom subclass implements:
  - `inputs_to_string(self, inputs, add_generation_prompt=True)`
  - `output_to_result(self, output, inputs=None)`

**Usage:**
```python
class _MyInputOutputProcessor(ModelDirectInputOutputProcessor):
    def inputs_to_string(self, inputs, add_generation_prompt=True):
        input_processor = get_input_processor(self._model_name)
        return input_processor.transform(inputs, add_generation_prompt)
    def output_to_result(self, output, inputs=None):
        output_processor = get_output_processor(self._model_name)
        return output_processor.transform(output, inputs)
```

---

### model_chat_with_thinking_separate_backend.py

**Purpose:**
Separates prompt construction and output parsing from model inference, using the OpenAI client directly and Granite IO processors for input/output.

**Key API Elements:**
- `get_input_processor`, `get_output_processor` (see above)
- `input_processor.transform(inputs, thinking=True)`
  - Converts structured chat input to a prompt string.
- `openai.OpenAI(base_url, api_key)`
  - Direct OpenAI API client (not part of granite-io).
- `output_processor.transform(outputs, inputs)`
  - Parses model output into structured response and reasoning.

**Usage:**
```python
prompt = input_processor.transform(ChatCompletionInputs(messages=messages, thinking=True))
result = openai_client.completions.create(model=..., prompt=prompt)
outputs = output_processor.transform(...)
```

---

### model_chat_with_citation_hallucination.py

**Purpose:**
Chat completion with both citation and hallucination detection enabled.

**Key API Elements:**
- `controls={"citations": True, "hallucinations": True}`
  - Enables both features in the output.
- `outputs.results[0].next_message.hallucinations`
  - List of detected hallucinations in the output.

**Usage:**
```python
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(
        messages=messages,
        documents=documents,
        controls={"citations": True, "hallucinations": True},
    )
)
if outputs.results[0].next_message.hallucinations:
    pprint.pprint(outputs.results[0].next_message.hallucinations)
```

---

### model_chat_with_mbrd_majority_voting.py

**Purpose:**
Chat completion with MBRD (Minimum Bayes Risk Decoding) and majority voting using ROUGE scoring.

**Key API Elements:**
- `MBRDMajorityVotingProcessor(base_processor)`
  - Wraps a base IO processor to add majority voting.
- `generate_inputs={"n": 20, "temperature": 0.6, "max_tokens": 1024}`
  - Requests multiple completions for voting.

**Usage:**
```python
voting_processor = MBRDMajorityVotingProcessor(base_processor)
results = voting_processor.create_chat_completion(completion_inputs)
```

---

### rerank_with_llm.py

**Purpose:**
Demonstrates a full RAG pipeline: retrieval, reranking, and chat completion using a local vLLM server and Granite 3.3 model.

**Key API Elements:**
- `LocalVLLMServer(model_name)`
  - Manages a local vLLM server subprocess.
- `server.make_backend()`
  - Returns a backend for the running server.
- `make_io_processor(model_name, backend)`
  - Creates an IO processor for the model.
- `Granite3Point3Inputs.model_validate({...})`
  - Validates and constructs structured input for Granite 3.3.
- `InMemoryRetriever(embeddings_location, embedding_model_name)`
  - In-memory vector database for retrieval.
- `RetrievalRequestProcessor(retriever, top_k)`
  - Augments chat input with retrieved documents.
- `RerankRequestProcessor(io_proc, rerank_top_k, return_top_k, verbose)`
  - Reranks retrieved documents using the LLM.

**Usage:**
```python
server = LocalVLLMServer(model_name)
backend = server.make_backend()
io_proc = make_io_processor(model_name, backend=backend)
retriever = InMemoryRetriever(embeddings_location, embedding_model_name)
rag_processor = RetrievalRequestProcessor(retriever, top_k=32)
rag_chat_input = rag_processor.process(chat_input)[0]
rerank_processor = RerankRequestProcessor(io_proc, rerank_top_k=32, return_top_k=5, verbose=True)
rerank_chat_input = rerank_processor.process(rag_chat_input)
rag_output = io_proc.create_chat_completion(rerank_chat_input)
```

---

### watsonx_litellm.py

**Purpose:**
Chat completion using the LiteLLM backend, supporting watsonx, Replicate, or Ollama models, with environment-based configuration.

**Key API Elements:**
- `make_backend("litellm", {"model_name": model_name})`
  - Creates a LiteLLM backend for the specified model.
- `make_io_processor(model_type, backend=...)`
  - Creates an IO processor for the model type.
- Environment variables (via `dotenv`) configure backend credentials.

**Usage:**
```python
from dotenv import load_dotenv
load_dotenv()
io_processor = make_io_processor(model_type, backend=make_backend("litellm", {"model_name": model_name}))
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages, thinking=True))
```