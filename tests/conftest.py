# SPDX-License-Identifier: Apache-2.0

# Common code shared by tests

# Hugging Face model path
#
# String that, if you pass it to the Transformers library, will resolve to Granite 3.2.
# This points to a local directory or a local HF cached model,
# because local_files_only=True is being used for tests.
# Use `huggingface-cli download` to get local models.
#
# GRANITE_3_2_MODEL_STR = "ibm-granite/granite-3.2-8b-instruct"
GRANITE_3_2_MODEL_STR = "ibm-granite/granite-3.2-2b-instruct"
# GRANITE_3_2_MODEL_STR = "ibm-granite/granite-3.2-8b-instruct-preview"


def pytest_addoption(parser):
    """Options to skip backends or tweak model names (unset to skip)"""

    parser.addoption("--hf_model", action="store", default=GRANITE_3_2_MODEL_STR)

    # Testing with ollama and 3.1 for now
    parser.addoption("--openai_model", action="store", default="granite3.1-dense:2b")
