# Campaign Provenance

BO Forge 3.1.0 can initialize a campaign with a versioned provenance manifest while
keeping the YAML configuration and CSV log as the campaign source data.

## Managed And Legacy Campaigns

Create a managed campaign with:

```bash
bo-forge init-log --config configs/my_campaign.yaml --log work/my_campaign.csv
```

The command creates both `work/my_campaign.csv` and
`work/my_campaign.csv.manifest.json`. It refuses to overwrite either file. Campaigns
created in Streamlit use the same initialization path.

Existing CSV campaigns without a manifest remain legacy campaigns. They load, suggest,
append, review, observe, report, and plot as before. BO Forge does not silently create a
manifest for them. Explicit legacy adoption is deferred to v3.1.2.

Manifest presence identifies a managed campaign. Keep the CSV and its manifest together
when moving, restoring, or backing up a campaign. A newly loaded CSV whose sidecar was
deleted or omitted is indistinguishable from a genuine legacy campaign in schema v1;
an already loaded managed session rejects sidecar disappearance before mutation.

Inspect either kind with:

```bash
bo-forge provenance --config configs/my_campaign.yaml --log work/my_campaign.csv
```

Python callers can use `campaign.provenance_summary()` or the top-level
`provenance_summary(config_path, log_path)` helper.

## Manifest Location And Format

The sidecar is stored beside the canonical log as `<log>.manifest.json`. Schema version
1 is UTF-8 JSON with stable key ordering, two-space indentation, and a final newline.
It records:

- an immutable UUID campaign ID and UTC creation/update timestamps;
- relative config and log references, never embedded author-home paths. A config under
  the author home uses a `~/...` reference when its manifest is outside that home;
- the exact config text, its byte SHA-256, and its normalized semantic SHA-256;
- the current log SHA-256 and row count;
- acquisition, model profile, seed, initial-design, sampling, restart, batch, distance,
  and applicable fidelity-optimizer settings;
- deduplicated process environment snapshots;
- an append-only mutation event ledger;
- an optional `pending_transaction` used for interrupted-write recovery.

The semantic config identity is built from the fully parsed configuration, including
defaults and ordered variables, objectives, and stages. Constraint and cost expressions
are represented by canonical expression syntax trees. Comments, YAML formatting, and
mapping-key order can change the byte hash without changing semantic identity. A real
configured-value change changes semantic identity.

## Event Ledger

Schema version 1 records these explicit operations:

- `initialize`;
- `append_suggestions`;
- `review_suggestion`;
- `mark_observed`.

Each event has a monotonic sequence, UUID, UTC timestamp, ordered affected row IDs,
previous/resulting log hashes, environment ID, and bounded operation metadata. Dry-run
suggestions, validation, summaries, reports, plots, and other reads do not create events.

Environment snapshots include BO Forge, Python, platform, direct scientific dependency
versions, and Git commit/dirty state when discoverable. Git discovery failures are
recorded as `unknown` and never block a campaign. Environments are deduplicated by a
canonical SHA-256 and are diagnostic snapshots, not complete reproducibility guarantees.

## Managed Mutations And Recovery

Append, review, and observation mutations use the existing canonical log lock. BO Forge
validates the current config, CSV, manifest, expected log fingerprint, and recorded
hashes before writing. It then records a pending transaction, atomically replaces and
post-validates the CSV, and finalizes the event.

Ordinary write or validation failures roll back the CSV and manifest together using
atomic replacement. If rollback cannot restore the CSV, BO Forge retains the recovery
backup and reports its path instead of deleting the last known-good bytes. If the process
is interrupted between replacements, the files can retain a pending transaction.
Read-only inspection reports that state but does not repair it. At the next managed
mutation BO Forge:

- finalizes the event when the CSV matches the intended resulting hash;
- cancels the pending transaction when the CSV matches the prior hash;
- fails closed without writing when the CSV matches neither hash.

Campaign load and managed mutation both fail with `LogConflictError` when current config
or log bytes no longer match the manifest. `bo-forge provenance` and the provenance API
remain read-only diagnostic paths: they report `mismatch` or `pending_recovery`, and the
CLI exits nonzero until the managed state is finalized and valid. `ProvenanceError` is
reserved for missing diagnostic inputs or unreadable, malformed, or unsupported manifest
data.

## Trust Boundary

The manifest is integrity and lineage metadata. It is not signed, tamper-proof, an
authorization mechanism, or a durable security audit log. Anyone who can modify the
campaign files can potentially modify the manifest as well. Continue to protect working
directories, back up campaign files, avoid unsupported multi-host writes, and follow the
trusted-deployment guidance in [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) and
[API_SECURITY.md](API_SECURITY.md).
