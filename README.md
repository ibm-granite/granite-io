# Granite IO Processing

## Introduction

Granite IO Processing is a framework which enables you to transform how a user calls or infers an IBM Granite model and how the output from the model is returned to the user. In other words, the framework allows you to extend the functionality of calling the model.

## Getting Started

### Requirements

* Python 3.10 to Python 3.11

### Installation

We recommend using a Python virtual environment with Python 3.10+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
```

> [!TIP]
> If you use [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or other such tools for Python version management, create the virtual environment with that tool instead of venv. Otherwise, you may have issues with installed packages not being found as they are linked to your Python version management tool and not `venv`.

You can install the Granite IO Processor as follows:

#### From Source

To install from source(GitHub Repository):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
git clone https://github.com/ibm-granite/granite-io
cd granite-io
pip install -e .
```

### Quick Start

Some sample code showing how to use the framework:

**Run using `granite3.2` in Ollama**
```py
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2"
io_processor = make_io_processor(
    model_name, backend=make_backend("openai", {"model_name": model_name})
)
messages=[
    UserMessage(
        content="What's the fastest way for a seller to visit all the cities in their region?",
    )
]

# Without Thinking
outputs = io_processor.create_chat_completion(ChatCompletionInputs(messages=messages))
print("-- NO THINKING ------")
print(outputs.next_message.content)

# With Thinking
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print("-- WITH THINKING ------")
print(">> Thoughts:")
print(outputs.reasoning_content)
print(">> Response:")
print(outputs.next_message.content)
```

### Try It Out!

To help you get up and running as quickly as possible with the Granite IO Processing framework, check out the following resources which demonstrate how to use the framework:

> [!TIP]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first.

- Python script examples (**require Ollama server running with Granite 3.2 pulled/cached**):
  - [Granite 3.2 chat request](./examples/inference.py)
  - [Granite 3.2 chat request with thinking](./examples/inference_with_thinking.py)
- Jupyter notebook tutorial:
  - [IO](./notebooks/io.ipynb)

## Contributing

Check out our [contributing guide](CONTRIBUTING.md) to learn how to contribute.
