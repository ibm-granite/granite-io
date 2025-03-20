# Granite IO Processing

## Introduction

Granite IO Processing is a framework which enables you to transform how a user calls or infers an IBM Granite model and how the output from the model is returned to the user. In other words, the framework allows you to extend the functionality of calling the model.

## Getting Started

### Requirements

* Python 3.10+

### Installation

We recommend using a Python virtual environment with Python 3.10+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
```

> [!TIP]
> If you use [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or other such tools for Python version management, create the virtual environment with that tool instead of venv. Otherwise, you may have issues with installed packages not being found as they are linked to your Python version management tool and not `venv`.

There are 2 ways to install the Granite IO Processor as follows:

#### From Release

To install from release ([PyPi package](https://pypi.org/project/granite-io/)):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
pip install granite-io
sudo python3 -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab
```

> [!NOTE]
> `granite-io` uses [NLTK Data](https://www.nltk.org/data.html) [Punkt Sentence Tokenizer](https://www.nltk.org/api/nltk.tokenize.punkt.html) for extracting contents when parsing output from a model. The command above shows how to install tokenizers for MacOS. Check out [install guide](https://www.nltk.org/install.html) for other OSes.

#### From Source

To install from source(GitHub Repository):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
git clone https://github.com/ibm-granite/granite-io
cd granite-io
pip install -e .
python3 sudo python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab
```

> [!NOTE]
> `granite-io` uses [NLTK Data](https://www.nltk.org/data.html) [Punkt Sentence Tokenizer](https://www.nltk.org/api/nltk.tokenize.punkt.html) for extracting contents when parsing output from a model. The command above shows how to install tokenizers for MacOS. Check out [install guide](https://www.nltk.org/install.html) for other OSes.

### Framework Example

Sample code snippet showing how to use the framework:

```py
from granite_io import make_backend, make_io_processor
from granite_io.types import ChatCompletionInputs, UserMessage

model_name = "granite3.2:8b"
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
print("------ WITHOUT THINKING ------")
print(outputs.results[0].next_message.content)

# With Thinking
outputs = io_processor.create_chat_completion(
    ChatCompletionInputs(messages=messages, thinking=True)
)
print("------ WITH THINKING ------")
print(">> Thoughts:")
print(outputs.results[0].next_message.reasoning_content)
print(">> Response:")
print(outputs.results[0].next_message.content)
```

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first.
> You will need additional packages to be able to run the OpenAI example. They can be installed by running `pip install -e "granite-io[openai]"`. Replace package name `granite-io` with `.` if installing from source.
>
> To be able to run the above code snippet, you will need an [Ollama](https://ollama.com/) server [running locally](https://github.com/ollama/ollama?tab=readme-ov-file#start-ollama) and [IBM Granite 3.2](https://www.ibm.com/granite) model cached (`ollama pull granite3.2:8b`).

### Try It Out!

To help you get up and running as quickly as possible with the Granite IO Processing framework, check out the following resources which demonstrate further how to use the framework:

1. Python script examples:

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first.
> You will need additional packages to be able to run the examples. They can be installed by running `pip install -e "granite-io[openai]"` and `pip install -e "granite-io[litellm]`. Replace package name `granite-io` with `.` if installing from source.
>
> You will also need an [Ollama](https://ollama.com/) server [running locally](https://github.com/ollama/ollama?tab=readme-ov-file#start-ollama) and [IBM Granite 3.2](https://www.ibm.com/new/announcements/ibm-granite-3-2-open-source-reasoning-and-vision) model cached (`ollama pull granite3.2:8b`).

   - [Granite 3.2 chat request](./examples/model_chat.py)
   - [Granite 3.2 chat request with thinking](./examples/inference_with_thinking.py)
   - [Using watsonx.ai](./examples/watsonx_litellm.py)

2. Jupyter notebook tutorials:

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first. You will also need additional packages to be able to run the Jupyter notebook. They can be installed by running `pip install -e "granite-io[transformers]"` and `pip install -e "granite-io[notebook]"`. Replace package name `granite-io` with `.` if installing from source. The notebooks can be then run with following command `jupyter notebook <path_to_notebook>`.

   - [IO](./notebooks/io.ipynb)

## Contributing

Check out our [contributing guide](CONTRIBUTING.md) to learn how to contribute.
