<div align="center">

<img src="./mkdocs/docs/img/ibm_logo_rebus.png" alt="IBM Logo" width="200"/>

# Granite IO Processing

[![PyPI version](https://img.shields.io/pypi/v/granite-io?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/granite-io/)
[![Python versions](https://img.shields.io/pypi/pyversions/granite-io?logo=python&logoColor=white)](https://pypi.org/project/granite-io/)
[![License](https://img.shields.io/github/license/ibm-granite/granite-io?color=green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/ibm-granite/granite-io?style=social)](https://github.com/ibm-granite/granite-io/stargazers)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue?logo=materialformkdocs)](./mkdocs/)
[![Downloads](https://img.shields.io/pypi/dm/granite-io?color=orange&logo=pypi)](https://pypi.org/project/granite-io/)

**🚀 A powerful framework for extending IBM Granite model functionality through input/output processing**

</div>

## Introduction

Granite IO Processing is a framework which enables you to transform how a user calls or infers an IBM Granite model and how the output from the model is returned to the user. In other words, the framework allows you to extend the functionality of calling the model.

## Getting Started

### 📋 Requirements

* Python 3.10+

### 💾 Installation

We recommend using a Python virtual environment with Python 3.10+. Here is how to setup a virtual environment using [Python venv](https://docs.python.org/3/library/venv.html):

```
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
```

> [!TIP]
> If you use [pyenv](https://github.com/pyenv/pyenv), [Conda Miniforge](https://github.com/conda-forge/miniforge) or other such tools for Python version management, create the virtual environment with that tool instead of venv. Otherwise, you may have issues with installed packages not being found as they are linked to your Python version management tool and not `venv`.

There are 2 ways to install the Granite IO Processor as follows:

#### 📦 From Release

To install from release ([PyPi package](https://pypi.org/project/granite-io/)):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
pip install granite-io
```

#### 🔧 From Source

To install from source(GitHub Repository):

```shell
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
git clone https://github.com/ibm-granite/granite-io
cd granite-io
pip install -e .
```

### 🎯 Framework Example

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

#### 📊 Sample Output

When you run the above example, you should see output similar to this:

```
$ python test.py
------ WITHOUT THINKING ------
The problem you're describing is a variant of the well-known Traveling Salesman Problem (TSP), which involves finding the shortest possible route that visits a given set of cities and returns to the origin. For a specific region, the optimal solution depends on the geographical layout and the distance between each city pair.

Here are some general strategies for solving this problem:

1. **Exact Algorithm**: Use mathematical optimization techniques like branch-and-bound or dynamic programming to find an exact solution if your dataset isn't too large. These methods can provide the shortest route, but they might be computationally expensive for a large number of cities.

2. **Approximation Algorithms**: For larger datasets, exact algorithms may become impractical due to computational complexity. In such cases, heuristic or approximation algorithms can provide near-optimal solutions more efficiently. Examples include the Nearest Neighbor algorithm, 2-opt swap, Lin–Kernighan–Helsgaun algorithm, and Christofides Algorithm (for metric TSP).

3. **Genetic Algorithms**: These are evolutionary computation techniques inspired by natural selection and genetics. Genetic algorithms can handle larger datasets and may find a near-optimal solution in less time than exact methods, though they do not guarantee optimality.

4. **Machine Learning/AI Approaches**: Machine learning models, such as deep reinforcement learning, have been developed to solve TSP, especially for large instances where traditional methods struggle.

Remember that the optimal method would depend on the number of cities (nodes) and their distribution, computational resources at hand, and the level of accuracy required. 

It's also worth considering additional constraints like time-of-day traffic, road conditions, fuel efficiency, etc., which might influence the route in real-world scenarios. These can be integrated into more advanced algorithms and models for a more practical, efficient solution.
------ WITH THINKING ------
>> Thoughts:
To solve this problem, we need to consider it as a variant of the Traveling Salesman Problem (TSP), which is an NP-hard problem in combinatorial optimization. This means there isn't a straightforward, quick solution for large datasets due to its computational complexity. However, there are several approaches and heuristics that can provide solutions, even if not optimal, in a reasonable amount of time for practical applications.

1. Identify the number of cities (or nodes) - knowing this will help in selecting an appropriate algorithm or method.
2. Consider the constraints: Is there a time limit? Are there specific routes that should be avoided?
3. Decide on the scale and resources available for computation, as some methods are more suited for smaller datasets or require significant computational power.
4. Explore various algorithms designed to solve TSP, like brute force, dynamic programming, nearest neighbor, genetic algorithms, or approximation algorithms.
>> Response:
The problem you're asking about—visiting all cities in a salesman's region exactly once and then returning to the origin—is essentially a variation of the classical Traveling Salesman Problem (TSP). Given its complexity, there isn't an outright "fastest" method that works optimally for every scenario, especially as the number of cities increases. However, several strategies can provide acceptable solutions within practical time constraints:

1. **Greedy Algorithm - Nearest Neighbor:**
   This is one of the simplest heuristics for TSP and can be executed relatively quickly even on large datasets. The salesman starts at a random city and at each step moves to the nearest unvisited city until all cities have been visited. While not guaranteed to find the shortest route, it provides a decent approximation, especially in larger regions where the impact of local suboptimality is lessened.

2. **Christofides Algorithm (for Euclidean planes):**
   If distances between cities form a metric that satisfies the triangle inequality and cities lie in a 2-dimensional Euclidean space, the Christofides algorithm can be used. This algorithm guarantees a solution within 1.5 times the optimal solution but is more computationally intensive than simpler methods.

3. **Genetic Algorithms:**
   These are powerful optimization techniques inspired by evolutionary biology. They work by iteratively improving upon a population of candidate solutions via processes like selection, crossover, and mutation. Genetic algorithms can provide high-quality solutions for TSP but require more computational resources and time compared to simpler heuristics.

4. **Linear Programming Heuristics:**
   Methods such as the one proposed by Lin and Kernighan (called "Karp's algorithm") or more recent linear programming formulations can yield near-optimal solutions, though they typically necessitate specialized software and significant computational power.

5. **Software Tools & APIs:**
   For immediate application needs without wanting to build a custom solution, leveraging existing tools like Google's OR-Tools is advisable. These platforms have robust implementations of various TSP algorithms that can be configured according to specific requirements (like time limits or distance metrics).

6. **Prioritize Nearby Cities Initially:**
   If geographical constraints make certain regions more accessible (e.g., due to better infrastructure), prioritizing visits within those nearby areas first could save time.

When choosing a method, consider factors like the number of cities, available computational resources, required solution quality, and the specific characteristics of your region (like whether distances follow a grid pattern or are more random). For small-scale problems or immediate application needs, simpler methods might suffice. Larger-scale or time-sensitive applications may benefit from more advanced algorithms or dedicated software solutions.
```

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first.
> You will need additional packages to be able to run the OpenAI example. They can be installed by running `pip install -e "granite-io[openai]"`. Replace package name `granite-io` with `.` if installing from source.
>
> To be able to run the above code snippet, you will need an [Ollama](https://ollama.com/) server [running locally](https://github.com/ollama/ollama?tab=readme-ov-file#start-ollama) and [IBM Granite 3.2](https://www.ibm.com/granite) model cached (`ollama pull granite3.2:8b`).

### Try It Out!

To help you get up and running as quickly as possible with the Granite IO Processing framework, check out the following resources which demonstrate further how to use the framework:

1. **Python script examples:**

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#installation) steps first.
> You will need additional packages to be able to run the examples. They can be installed by running `pip install -e "granite-io[openai]"` and `pip install -e "granite-io[litellm]`. Replace package name `granite-io` with `.` if installing from source.
>
> You will also need an [Ollama](https://ollama.com/) server [running locally](https://github.com/ollama/ollama?tab=readme-ov-file#start-ollama) and [IBM Granite 3.2](https://www.ibm.com/new/announcements/ibm-granite-3-2-open-source-reasoning-and-vision) model cached (`ollama pull granite3.2:8b`).

   - 💬 [Granite 3.2 chat request](./examples/model_chat.py)
   - 🧠 [Granite 3.2 chat request with thinking](./examples/inference_with_thinking.py)
   - 📚 [Granite 3.2 RAG](./examples/model_chat_with_citation.py)
   - 🚨 [Granite 3.2 RAG and hallucinations](./examples/model_chat_with_citation_hallucination.py)
   - 🗳️ [Granite 3.2 MBRD majority voting](./examples/model_chat_with_mbrd_majority_voting.py)
   - 🔧 [Granite 3.2 custom IO processor](./examples/model_chat_with_thinking_custom_io_processor.py)
   - 🔀 [Granite 3.2 separate input and out processors](./examples/model_chat_with_thinking_separate_backend.py)
   - ☁️ [Using watsonx.ai](./examples/watsonx_litellm.py)

2. **Jupyter notebook tutorials:**

> [!IMPORTANT]
> To get started with the examples, make sure you have followed the [Installation](#💾-installation) steps first. You will also need additional packages to be able to run the Jupyter notebook. They can be installed by running `pip install -e "granite-io[transformers]"` and `pip install -e "granite-io[notebook]"`. Replace package name `granite-io` with `.` if installing from source. The notebooks can be then run with following command `jupyter notebook <path_to_notebook>`.

   - 🤔 [Granite 3.2 answerability](./notebooks/answerability.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/answerability.ipynb)
   - 🎯 [Granite 3.2 certainity](./notebooks/certainty.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/certainty.ipynb)
   - 📖 [Granite 3.2 citations](./notebooks/citations.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/citations.ipynb)
   - 🚨 [Granite 3.2 hallucinations](./notebooks/hallucinations.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/hallucinations.ipynb)
   - 🔄 [Granite 3.2 IO APIs](./notebooks/io.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/io.ipynb)
   - ✏️ [Granite 3.2 query rewrite](./notebooks/query_rewrite.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/query_rewrite.ipynb)
   - 📚 [Granite 3.2 RAG](./notebooks/rag.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/rag.ipynb)
   - 🔍 [Granite 3.2 retrieval](./notebooks/retrieval.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/retrieval.ipynb)
   - 🗳️ [Granite 3.2 simple majority voting](./notebooks/voting.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ibm-granite/granite-io/blob/main/notebooks/voting.ipynb)


## Documentation and Architecture

This project uses [MkDocs](https://www.mkdocs.org/) with the Material theme for comprehensive documentation. To access and work with the documentation:

- **📁 Documentation source**: Located in [`./mkdocs/docs/`](./mkdocs/docs/)
- **⚙️ Configuration**: See [`./mkdocs/mkdocs.yml`](./mkdocs/mkdocs.yml)
- **🚀 Local development**: Follow the instructions in [`./mkdocs/README.md`](./mkdocs/README.md)

To serve the documentation locally:
```bash
cd mkdocs
pip install mkdocs mkdocs-material
mkdocs serve
```

<div align="center">

Click the badge below to view the documentation site:

[![Documentation](https://img.shields.io/badge/_View_Documentation_Site-4285F4?style=for-the-badge&logo=googledocs&logoColor=white)](https://ibm-granite.github.io/granite-io/)

</div>

---

## 🤝 Contributing

<div align="center">

Click below to view our contributing guide:

[![Contributing](https://img.shields.io/badge/_Contributing_Guide-28a745?style=for-the-badge&logo=github&logoColor=white)](CONTRIBUTING.md)

</div>
