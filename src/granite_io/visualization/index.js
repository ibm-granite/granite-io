// Arrows for expanding/collapsing citations
let expand_text = "&#x25BC;";
let collapse_text = "&#x25B2;"

/**
* Given a citation ID, create an HTML element of class "citation-num"
* displaying the citation ID
*/
function show_single_citation_num(citation_id) {
    let single_citation_num = document.createElement("span");
    single_citation_num.classList.add("citation-num");
    single_citation_num.setAttribute("citation_id", citation_id);
    single_citation_num.innerHTML = citation_id;
    
    return single_citation_num;
}

/**
* Given a citation ID, create an HTML element of class "citation-ref"
* acting as a reference to the citation with the given ID. The resulting
* element is an HTML element of class "citation-num" enriched with the
* additional "citation-ref" class to enable the element to act as a
* reference on mouseover/mouseout/click events.
*/
function show_single_citation_ref(citation_id) {
    let single_citation_ref = show_single_citation_num(citation_id);
    single_citation_ref.classList.add("citation-ref");
    
    return single_citation_ref;
}

/**
* Given the ID and context text of a citation, create an HTML element of
* class "citation" displaying the citation (incl. its ID and context text)
*/
function show_single_citation(citation_id, citation_text) {
    let single_citation_div = document.createElement("div");
    single_citation_div.classList.add("citation");
    single_citation_div.setAttribute("citation_id", citation_id);
    
    let single_citation_num = show_single_citation_num(citation_id);
    single_citation_div.appendChild(single_citation_num);
    
    let single_citation_text = document.createElement("span");
    single_citation_text.classList.add("citation-text");
    single_citation_text.innerHTML = citation_text;
    single_citation_div.appendChild(single_citation_text);
    
    let single_citation_arrow = document.createElement("span");
    single_citation_arrow.classList.add("citation-arrow")
    single_citation_arrow.innerHTML = expand_text;
    single_citation_div.appendChild(single_citation_arrow);
    
    return single_citation_div;
}

/**
* Given the ID of a citation, the text of the passage where the citation
* appears and the begin/end span where the citation appears within the passage,
* create an HTML element of class "citation-doc" displaying the passage
* with the citation text highlighted. The citation span is expected to
* be passed as an array of the form of [span_begin, span_end].
*/
function show_single_citation_doc(citation_id, doc_text, span_to_highlight) {
    let single_citation_doc = document.createElement("div");
    single_citation_doc.classList.add("citation-doc");
    single_citation_doc.setAttribute("citation_id", citation_id);
    single_citation_doc.innerHTML = highlight_span_in_doc(doc_text, span_to_highlight);
    
    return single_citation_doc;
}    

/**
* Given the text of a passage and the span to highlight within that passage,
* return a string containing the text of the passage with the span
* highlighted. The span is expected to be passed as an array of the form
* [span_begin, span_end]
*/
function highlight_span_in_doc(doc_text, span_to_highlight) {
    let span_begin = span_to_highlight[0];
    let span_end = span_to_highlight[1];
        
    let doc_with_highlights = "";
    

    if (span_begin > 0) {
        doc_with_highlights += doc_text.substring(0, span_begin);
    }
            
    doc_with_highlights += "<span class='doc-highlight'>" + doc_text.substring(span_begin, span_end) + "</span>";

        
    if (span_end < doc_text.length) {
        doc_with_highlights += doc_text.substring(span_end, doc_text.length-1);
    }
        
    return doc_with_highlights;
}

/**
* Given a set of citations, group together citations corresponding to the
* same response span by grouping the flat citation list by response_begin.
* Also sort the resulting groups by response_begin.
*/
function group_citations_by_response_span(citations) {
    let citations_by_response_span = Object.groupBy(citations, ({ response_begin }) => response_begin);
    let citations_by_response_span_list = Object.keys(citations_by_response_span).map(key => ({
        response_begin: parseInt(key),
        response_end: citations_by_response_span[key][0]["response_end"],
        citations: citations_by_response_span[key]}));
    
    let citations_by_response_span_sorted = citations_by_response_span_list.sort(function(a,b) {return a["response_begin"] - b["response_begin"];});
    
    return citations_by_response_span_sorted
    
}

/**
* Given a response, a set of citations, and a set of passages, create
* and return an HTML element that corresponds to the visualization of the
* response with inline citations
*/
function show_inline_citations(response, citations, docs) {
    
    // Group and sort citations by response_span
    let citations_by_response_span_sorted = group_citations_by_response_span(citations);
    
    // Iterate over citation groups and populate two HTML strings:
    // response_html: Holding response with references to corresponding citations
    // citations_html: Holding citation context text and its visualization within the corresponding passage
    let response_html = "";
    let citations_html = "";       
    let cur_response_idx = 0;
    let cur_citation_id = 1;
        
    for (const cgroup of citations_by_response_span_sorted) {
        let cgroup_begin = cgroup["response_begin"];
        let cgroup_end = cgroup["response_end"];
          
          
        if (cgroup_begin > cur_response_idx) {
            response_html += response.substring(cur_response_idx, cgroup_begin);
        }
          
        response_html += response.substring(cgroup_begin, cgroup_end);
        
        for (const c of cgroup["citations"]) {     
            response_html += show_single_citation_ref(cur_citation_id).outerHTML;
            citations_html += show_single_citation(cur_citation_id, c["context_text"]).outerHTML;
            citations_html += show_single_citation_doc(cur_citation_id, docs[c["doc_id"]]["text"], [c["context_begin"], c["context_end"]]).outerHTML;
            cur_citation_id++;
        }
        cur_response_idx = cgroup_end;
    }
        
    if (cur_response_idx < response.length) {
        response_html += response.substring(cur_response_idx, response.length-1);
    }
    
    // Assemble the HTML strings into the final element
    let container_div = document.createElement("div");
    
    let response_div = document.createElement("div");
    response_div.innerHTML = response_html;
    response_div.classList.add("response");
    container_div.appendChild(response_div);
      
    let citations_div = document.createElement("div");
    citations_div.innerHTML = citations_html;
    citations_div.classList.add("refs");
    container_div.appendChild(citations_div);
    
    return container_div;
}


/**
* When a citation reference is clicked, programmatically click
* the corresponding citation
*/
function handle_citation_ref_click(evt, container_div) {
    let citation_ref_obj = evt.currentTarget;
    let citation_id = citation_ref_obj.getAttribute("citation_id");
    let citation_obj = container_div.querySelector(".citation[citation_id='" + citation_id + "']");
    citation_obj.click();
}

/**
* When the mouse is over a citation reference, highlight both the citation
* reference and the corresponding citation
*/
function handle_citation_ref_mouseover(evt, container_div) {
    let citation_ref_obj = evt.currentTarget; 
    let citation_id = citation_ref_obj.getAttribute("citation_id");
    let citation_obj = container_div.querySelector(".citation[citation_id='" + citation_id + "']");
    citation_obj.classList.add("citation-highlighted");
    citation_ref_obj.classList.add("citation-highlighted");
}

/**
* When the mouse stops being over a citation reference, remove the highlight
* from the citation reference and the corresponding citation
*/
function handle_citation_ref_mouseout(evt, container_div) {
    let citation_ref_obj = evt.currentTarget;
    let citation_id = citation_ref_obj.getAttribute("citation_id");
    let citation_obj = container_div.querySelector(".citation[citation_id='" + citation_id + "']");
    citation_obj.classList.remove("citation-highlighted");
    citation_ref_obj.classList.remove("citation-highlighted");
}

/**
* When a citation is clicked, toggle the visibility of the element showing this
* citation within its document context and appropriately modify the
* expand/collapse arrow within the citation element
*/ 
function handle_citation_click(evt, container_div) {
    let citation_id = evt.currentTarget.getAttribute("citation_id");
    let citation_doc_obj = container_div.querySelector(".citation-doc[citation_id='" + citation_id + "']");
    if (citation_doc_obj.style.display == "block") {
        citation_doc_obj.style.display = "none";
        evt.currentTarget.getElementsByClassName("citation-arrow")[0].innerHTML = expand_text;
    } else {
        citation_doc_obj.style.display = "block";
        evt.currentTarget.getElementsByClassName("citation-arrow")[0].innerHTML = collapse_text;
    }
}

/**
* When the mouse is over a citation, highlight both the citation and the
* corresponding citation reference
*/
function handle_citation_mouseover(evt, container_div) {
    let citation_obj = evt.currentTarget; 
    let citation_id = citation_obj.getAttribute("citation_id");
    let citation_ref_obj = container_div.querySelector(".citation-ref[citation_id='" + citation_id + "']");
    citation_ref_obj.classList.add("citation-highlighted");
    citation_obj.classList.add("citation-highlighted");
}

/**
* When the mouse stops being over a citation, remove the highlight from the
* citation and the corresponding citation reference
*/
function handle_citation_mouseout(evt, container_div) {
    let citation_obj = evt.currentTarget;
    let citation_id = citation_obj.getAttribute("citation_id");
    let citation_ref_obj = container_div.querySelector(".citation-ref[citation_id='" + citation_id + "']");
    citation_ref_obj.classList.remove("citation-highlighted");
    citation_obj.classList.remove("citation-highlighted");
}

/**
* Rendering function for anywidget
*/
function render({ model, el }) {
        
    // Read input variables
    let documents = model.get("documents");
    let response = model.get("response");
    let citations = model.get("citations");
        
    // Create the citation visualization
    let citation_vis = show_inline_citations(response, citations, documents);    
    el.appendChild(citation_vis);
    el.classList.add("citations-widget");
      
    // Register event listeners for citation references
    let citation_ref_objs = el.getElementsByClassName("citation-ref");
    for (const c of citation_ref_objs) {
        c.addEventListener('click', (evt) => { handle_citation_ref_click(evt, el); });
        c.addEventListener('mouseover', (evt) => { handle_citation_ref_mouseover(evt, el); });
        c.addEventListener('mouseout', (evt) => { handle_citation_ref_mouseout(evt, el); });
    }
      
    // Register event listeners for citations
    let citation_objs = el.getElementsByClassName("citation");
    for (const c of citation_objs) {
        c.addEventListener('click', (evt) => { handle_citation_click(evt, el); });
        c.addEventListener('mouseover', (evt) => { handle_citation_mouseover(evt, el); });
        c.addEventListener('mouseout', (evt) => { handle_citation_mouseout(evt, el); });
    }
      
}

export default { render };