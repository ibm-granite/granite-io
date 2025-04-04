# Prompt Description Language (PDL) I/O Processor

[Prompt Description Language (PDL)](https://ibm.github.io/prompt-declaration-language/) is a declarative language designed for developers to create reliable, composable LLM prompts.

This contribution provides two I/O processors:
- a generic I/O processor that allows the logic of the processor to be implemented as a PDL function,
- a sequential scaling processor that given a prompt and a validator is automatically iterating with an underlining processor until the answer pass the validator.

