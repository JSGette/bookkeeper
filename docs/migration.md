# Database migration, validation, and rollback

The migration command creates a separate schema-v1 SQLite database. It does not rename, delete, or overwrite the original database or flat index. Cutover is manual. This runbook documents implemented tooling; no migration of a live personal database has been performed as part of the implementation.

Do not run old and new binaries against the same database. Mixed-version writers and live cutover are unsupported. Keep all writers stopped from inventory through validation and cutover, including HTTP daemons, stdio processes in editor windows, transcript watchers, indexing scripts, and any other process that can update either database.

## 1. Identify and freeze the source

Record the old executable/version, environment, configuration, database location, source directories, and transcript directories. Preserve the old installation and configuration so they can be restored independently of the new installation. Use an explicit database path rather than relying on a default cache directory.

For a managed daemon:

```bash
bookkeeper daemon status
bookkeeper daemon stop
```

Also stop per-window stdio servers and any new test server. A stopped daemon alone does not stop those processes. Prevent automatic restarts while migrating. Retain an offline backup of the stopped source, including any SQLite WAL sidecars; do not copy only the main `.db` file while a writer is active. The migrator uses SQLite's backup API to include committed WAL content.

Supported sources:

| Input | Selection and preservation |
|---|---|
| Legacy Bookkeeper SQLite | Pass the exact `.db` path. Original table/column values, IDs, vector BLOBs, and unknown columns/tables are inventoried and checked in the copy. New evidence/schema metadata is additive |
| Directory containing one SQLite database | Recognizes `bookkeeper.db` or `agents-rag.db`; use the exact file when the directory is ambiguous |
| Legacy flat index directory | Requires `index.npz` plus `chunks.json`; reads optional `texts.json`, `state.json`, `deprecated.json`, and `transcripts/*.state.json` |

If a directory contains multiple SQLite candidates or both a database and flat index, selection fails instead of guessing. For a flat index in an ambiguous directory, make a separate stopped snapshot containing the intended flat artifacts and pass that directory.

For flat inputs, recognized original artifacts are archived byte-for-byte in the destination's `legacy_artifacts` table, including unknown JSON fields and conflicting legacy text. A position map preserves the association between each imported chunk and its vector. Active document content reconstructed from chunks is labeled `reconstructed_chunks`; the original files remain unchanged. Unrecognized files are not imported automatically, so retain the whole original snapshot.

## 2. Inspect without creating a destination

Use absolute paths in real commands. `OLD` below is the selected file or flat snapshot directory; `NEW` must name a database that does not already exist.

```bash
bookkeeper migrate --from /path/to/old.db --to /path/to/bookkeeper-v1.db --dry-run
```

For a flat snapshot:

```bash
bookkeeper migrate --from /path/to/flat-snapshot --to /path/to/bookkeeper-v1.db --dry-run
```

The JSON report should say `status: "dry_run"`, identify the expected source/destination and format, and include a source manifest/fingerprint. SQLite inventory includes every user table's columns, row count, and content hash. Flat inventory includes recognized artifact hashes and vector/metadata dimensions.

Inspection rejects unsupported future schemas, corrupt SQLite integrity, invalid vector data, and incompatible flat-file shapes. It does not infer missing vectors, discard broken rows, load embedding models, or call Ollama. Investigate rejected input in a separate copy; do not repair the original just to bypass a validation error.

The dry run validates inputs, not available disk capacity or eventual model quality. Reserve space for the original, destination, staging files, and new evidence metadata. Keep the reported manifest with the migration record.

## 3. Copy and validate

With writers still stopped:

```bash
bookkeeper migrate --from /path/to/old.db --to /path/to/bookkeeper-v1.db
```

The command creates a destination-specific lock, a `.manifest.json` sidecar, and a `.staging` database. It copies SQLite through the backup API or imports flat artifacts into staging, applies the new schema there, and validates preservation, SQLite integrity, and foreign keys. It checks that the source fingerprint did not change during the operation.

Only a validated staging database is published at the final destination, without replacing an existing destination. The successful report says `status: "validated"` and `cutover: "manual"`. The original remains untouched. The destination's derived HNSW cache is cleared so it can be rebuilt against the final vector/evidence mapping; this is an intentional cache reset, not loss of source evidence or vector BLOBs.

The lock file may remain after success; its presence alone does not mean a migration is running. Do not start a server against `.staging` or edit its manifest.

## 4. Resume an interrupted migration

Keep the original inputs unchanged and use the same source/destination pair:

```bash
bookkeeper migrate --from /path/to/old.db --to /path/to/bookkeeper-v1.db --resume
```

Resume requires a matching saved manifest and revalidates preservation. Committed flat-document import units are reused; partially committed work is not treated as complete. If the source changed, or you selected a different source, choose a new destination after freezing it again. Do not remove or alter the manifest to force a resume.

If the final destination already exists, the command refuses to overwrite it, including with `--resume`. Inspect the prior successful report and destination rather than attempting to rerun migration over it.

## 5. Validate evidence before enabling writers

Review the migration report against the original inventory. The preservation checks establish that available legacy data survived the copy; they cannot recover information that was never stored.

- Legacy session summaries remain `legacy_summary`, and other uncertain legacy content remains explicitly incomplete/unknown. A summary cannot recover the full transcript. Re-ingest the original transcript if it still exists; otherwise retain the completeness label.
- Legacy vector bytes are retained, but an unknown embedding model/revision is not assumed compatible with the current query model. Those records remain available through lexical search. Reindex available source files and re-ingest raw transcripts to generate known compatible embeddings after cutover. Missing originals cannot be reconstructed from vectors.
- Summaries and reusable memory assertions are not promoted to verified facts. Evidence IDs identify stored revisions; callers must still examine source currency and deprecation.

Run the read-only evaluator on the new database with a user-labeled case file:

```bash
python benchmark.py --db /path/to/bookkeeper-v1.db --cases /path/to/labeled-cases.jsonl --config /path/to/new-config.yaml --budgets 1000 2000 4000 --output /path/to/evaluation-lexical.json --lexical-only
```

Run from the repository checkout with its dependencies installed. The evaluator requires schema v1 and never migrates the selected database. It opens a read-only snapshot and defaults to lexical retrieval with no model calls.

The JSONL format is one labeled object per line:

| Field | Meaning |
|---|---|
| `query` | Required nonempty search query, preserved exactly |
| `expected_paths` | Required list of relevant document paths; use an empty list for unanswerable cases |
| `source` | Optional source scope applied to retrieval and labeled-document lookup |
| `required_evidence` | Optional list of exact, case-sensitive source substrings that should survive retrieval/packing |
| `unanswerable` | Optional boolean; cannot be combined with positive path/evidence labels |
| `category` | Optional evaluation slice, such as a category from your own held-out corpus |

Choose labels from real tasks and source passages. Keep threshold-tuning examples separate from held-out acceptance cases. Include missing answers, version-sensitive queries, changed/deprecated decisions, late-session facts, and conflicting evidence when they occur in the corpus. No production-labeled dataset is bundled or implied.

Reports provide path recall, raw-evidence substring coverage before/after packing, and coverage in the current labeled source documents independently of retrieval. This last measurement helps distinguish retrieval failure from missing source data, but is not an oracle for an unspecified historical revision. Generated summaries are excluded from source-support counts.

Token counts measure the entire compact `serialize_packet` response, including metadata and escaping, using `cl100k_base`. A downstream model may use a different tokenizer. Reports also include budget failures, setup latency, retrieval/packing/total p50 and p95, category aggregates, and corpus/settings/model fingerprints. Retrieval runs once per case and is reused across budgets. The abstention score is explicitly a heuristic based on missing evidence or weak retrieval, not measured answer correctness.

After compatible embeddings exist, an explicit model-enabled comparison is available:

```bash
python benchmark.py --db /path/to/bookkeeper-v1.db --cases /path/to/labeled-cases.jsonl --config /path/to/new-config.yaml --budgets 1000 2000 4000 --output /path/to/evaluation-local-models.json --configured-models
```

This mode uses configured reranker settings and a matching known embedding profile. Set `EMBEDDING_MODEL` and `EMBEDDING_REVISION` consistently with the serving process; the legacy YAML `model` field is ignored by both server and evaluator. It uses cached local models and refuses an unknown/incompatible embedding identity; it does not download missing models or call generative/cloud LLMs. A failed reranker can still fall back to RRF, recorded in search metadata. Do not interpret a model-disabled run as the production dense/reranked result.

## 6. Cut over explicitly

Preserve the old configuration. Create a new configuration selecting the validated destination:

```yaml
db_path: /absolute/path/to/bookkeeper-v1.db
cache_dir: /absolute/path/to/new-bookkeeper-cache
```

On Windows, YAML paths such as `C:/Users/name/bookkeeper/bookkeeper-v1.db` avoid backslash escaping. Retain the intended source names/roots and transcript directories; memory scope and evidence lookup depend on them. Point `AGENTS_RAG_CONFIG` in the actual daemon or MCP launch environment to this new file. Merely editing a different shell's environment does not reconfigure an already installed service.

Start only the new binary against the new database. Verify the MCP `status()` output reports the intended database/schema, expected source counts, and pending ingestion count. Inspect representative `query_knowledge_base` results and resolve their individual `evidence_spans[].evidence_id` through `get_evidence`; check revision, completeness, and historical status. `get_full_document` is a current-source read and does not attest the exact retrieved revision.

Use `reindex(source)` for configured files and `distill_transcripts()` for original transcript directories when ready to regenerate derived data. These operations may load local models and create new revisions. Check asynchronous completion/status rather than treating a queued response as success. Use `retry_ingestion()` to queue failed/pending transcript jobs after fixing the underlying cause. Cursor checkpoints advance with durable evidence persistence.

The server startup, background watcher, and normal MCP requests can write audit/state data. Treat the first new-server run as the start of post-cutover writes, even if you have not intentionally saved a new memory yet. Keep the original frozen through your acceptance window.

## 7. Roll back without losing new work

**Before any new writes:** stop the new server, restore the original executable/environment and configuration, and point them at the unchanged original database. Keep the validated new copy and migration reports for diagnosis. Do not point the old binary at the new schema.

**After any new writes:** stop writers and retain the new database and its committed WAL state. Returning to the old snapshot alone would omit newer evidence, memory revisions, checkpoints, and audit data. A lossless downgrade exporter is not implemented. Recovery requires a separately designed and validated reconciliation/export of those writes; do not describe a pointer switch as lossless rollback.

There is no automatic dual-write, live synchronization, or mixed-version compatibility layer. Never merge SQLite files by overwriting the old database with the new one or allowing both binaries to write the same path. Original data, new data, and the validation record remain separate until a deliberate recovery or retention decision is made.
