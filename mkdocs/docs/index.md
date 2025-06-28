# IBM Granite IO

<a href="https://research.ibm.com/" target="_blank"><img src="./img/ibm_logo_rebus.png" alt="IBM Logo" width="300" style="display:block; margin:auto;"/></a>

## Introduction

IBM Granite IO is a framework that enables you to transform how a user calls or infers an IBM Granite model and how the output from the model is returned. The framework allows you to extend and customize the functionality of model inference and output processing.

---

## Quickstart

### Requirements

- Python 3.10+

### Installation

We recommend using a Python virtual environment:

```sh
python3 -m venv granite_io_venv
source granite_io_venv/bin/activate
```

Install Granite IO Processing from PyPI:

```sh
pip install granite-io
```

Or, to install from source:

```sh
git clone https://github.com/ibm-granite/granite-io
cd granite-io
pip install -e .
```

### Example: Using Granite IO Processing

```python
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

#### Sample Output

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

> **Note:**

> - You may need additional dependencies for some backends (e.g., OpenAI, LiteLLM). Install with `pip install -e "granite-io[openai]"` or similar as needed.
> - For local inference, you will need an [Ollama](https://ollama.com/) server running and the [IBM Granite 3.2](https://www.ibm.com/granite) model cached (`ollama pull granite3.2:8b`).

---

## Architecture

Granite IO Processing is designed to be modular and extensible. The main components are:

- **Backend:** Abstraction for different runtime backends for serving models for inference.
- **Input Processor:** Transforms a chat request prior to sending it to the model.
- **Output Processor:** Transforms output of a chat request from the model.
- **Input/Output (IO) Processor:** Combines both an input and output processor, and (optionally) a backend.

For a detailed explanation and diagrams, see the [Architecture & Design](design.md) page.

---

## Contributing

We welcome contributions! Check out our [contributing guide](https://github.com/ibm-granite/granite-io/blob/main/CONTRIBUTING.md) to learn how to get started.

---

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).