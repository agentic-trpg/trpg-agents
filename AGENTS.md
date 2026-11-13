# Repository guidance

## Optional local code graph

- For call chains, consumers, or cross-module impact, consider the local `trpg-agents` graph when available. Check the workspace, index location, and configured checkout against the revision under investigation, then confirm the project with `lct_graph.list_projects`.
- Use `lct_graph.resolve` and `definition` for symbols, and `callers` or `callees` for relationships. Verify consequential results in source. Use local source search for text lookups or when the matching graph is unavailable.
- Follow the `code-graph-research` skill for index maintenance. After code lands on `main`, verify the resulting revision and incrementally refresh this repository's local baseline graph if it exists. Keep candidate indexes separate from the baseline. First-time indexing and full rebuilds require explicit human authorization.
