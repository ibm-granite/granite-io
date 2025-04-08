# Granite I/O Backends

## openai

The `openai` backend works with models being served by an OpenAI API compatible provider.

### Environment variables

This backend supports the following settings with environment variables:

| --- | --- | --- |
| Env variable | Description | Example |
| --- | --- | --- |
| OPENAI_BASE_URL | OPENAI_BASE_URL | OPENAI_BASE_URL=http://localhost:8000/v1  |

## litellm

The `litellm` backend works with models being served by an LiteLLM supported provider.

## transformers

The `transformers` backend uses the transformers library to load a model which is then used as the backend provider.
