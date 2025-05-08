# SPDX-License-Identifier: Apache-2.0

"""
Widget for visualizing citations.
"""

import anywidget
import traitlets
import pathlib

# Local
from granite_io.types import (
    ChatCompletionInputs,
    ChatCompletionResults,
)

class CitationsWidget():
    def __init__(self):
        super().__init__()
             
    def show(self, inputs: ChatCompletionInputs, outputs: ChatCompletionResults):
        documents = [ doc.model_dump(mode='json') for doc in inputs.documents ]
        response = outputs.results[0].next_message.content
        citations = [ citation.model_dump(mode='json') for citation in outputs.results[0].next_message.citations ]
        
        return CitationsWidgetInstance(response=response, documents=documents, citations=citations)
    
    
class CitationsWidgetInstance(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent/"index.js"
    _css = pathlib.Path(__file__).parent/"index.css"
    
    response = traitlets.Unicode({}).tag(sync=True)
    documents = traitlets.List([]).tag(sync=True)
    citations = traitlets.List([]).tag(sync=True)