# Frontend Implementation Plan

This plan describes how to build an explainable knowledge-base UI on top of the
current ResearchAgent project.

## Goal

Build a frontend that supports keyword retrieval and later grows into an
explainable paper QA system. The UI should show not only the final answer, but
also the retrieved evidence, selected context, token usage, and experiment
results.

## Product Direction

The first version should be an internal research tool, not a generic chatbot.
The core value is:

```text
query -> retrieval evidence -> selected context -> answer -> metrics
```

This makes retrieval and context selection behavior inspectable during
experiments.

## Recommended Architecture

Use a lightweight API backend between the frontend and the existing Python
pipeline.

```text
Frontend UI
  -> FastAPI backend
    -> storage/metadata.json keyword search
    -> FAISS semantic retrieval
    -> app.answer_query for QA
    -> experiments/*.csv for result viewing
```

Recommended frontend stack:

```text
React + Vite + TypeScript
Ant Design
```

Ant Design is a good fit because this is an experiment/workbench UI with forms,
tables, filters, and dense result inspection.

## Implementation Order

### 1. Backend Keyword Search API

Add a FastAPI backend with a keyword search endpoint:

```text
POST /search
```

Initial behavior:

- Load `storage/metadata.json`.
- Search over `text`, `source`, `page`, and optionally paper title.
- Return matching chunks with keyword highlights.
- Include `source`, `page`, `score`, `chunk_id`, and text preview.

This step should not call the LLM. It is only for validating retrieval and
corpus inspection.

### 2. Frontend Keyword Search Page

Create the first UI page:

```text
Keyword input
Search button
Result list/table
Highlighted matches
Source and page metadata
```

Useful controls:

- keyword query
- max results
- source filter
- case-sensitive toggle if needed

This page is the foundation for inspecting whether the knowledge base content is
indexed and searchable.

### 3. QA API

Add an ask endpoint:

```text
POST /ask
```

It should call the existing `answer_query` pipeline and return:

- final answer
- retrieved chunks
- selected chunks
- context tokens
- prompt/completion/total token usage if available
- details path or run id

Expose only safe parameters first:

- `strategy`
- `top_k`
- `context_budget`
- `compression`
- `compression_stage`

### 4. Explainable QA Page

Create a QA page with three work areas:

```text
Left: query and parameters
Center: answer
Right: retrieved evidence and selected context
```

The evidence panel should show:

- retrieved chunks
- selected chunks
- source file
- page
- score
- token estimate
- text preview

This is more useful than a simple chat UI because it reveals why an answer was
or was not supported by the context.

### 5. Experiment Results Page

Add a page that reads experiment CSV files such as:

```text
experiments/qa_v1_densex_summary.csv
experiments/qa_parent_v1_densex_results.csv
```

Display sortable/filterable tables for:

- `granularity`
- `budget`
- `answer_f1`
- `answer_recall`
- `context_tokens`
- `selected_gold_recall`
- `selected_relevance_precision`
- `token_efficiency`

This page avoids repeatedly opening CSV files manually.

### 6. Hybrid Retrieval And Advanced Modes

After the basic keyword and QA pages work, add advanced retrieval modes:

```text
keyword
semantic
hybrid
sentence-to-chunk
proposition-to-chunk
```

Hybrid retrieval can combine:

```text
keyword score + embedding similarity score
```

The fine-to-chunk modes should reuse the current parent aggregation logic:

```text
sentence/proposition retrieve -> parent chunk aggregation -> context selection
```

### 7. Evaluation Overlay

For questions that exist in `evaluation/questions.jsonl`, optionally show
evaluation fields after a run:

- matched gold items
- selected gold recall
- selected relevance precision
- answer F1 / recall

This should be treated as an experiment/debug feature, not a user-facing answer
quality guarantee.

## First Milestone

The first usable milestone should include:

```text
FastAPI /search
React keyword search page
highlighted chunk results
source/page display
```

Do not start with a chat page. Keyword search is simpler, easier to validate,
and directly useful for checking whether the corpus contains the expected
evidence.

## Second Milestone

Add:

```text
POST /ask
QA page
retrieved/selected context panels
token usage display
```

At this point the UI becomes an explainable knowledge-base QA system.

## Third Milestone

Add:

```text
experiment result CSV viewer
granularity comparison tables
fine-to-chunk mode controls
```

This turns the UI into a practical experiment dashboard for the current
research workflow.
