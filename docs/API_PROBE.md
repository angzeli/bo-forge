# Experimental FastAPI Probe

BO Forge includes an experimental optional FastAPI probe around the internal
`CampaignAppService`. It is for local or trusted-network API exploration only.
It is not a stable public API and does not replace the Streamlit workbench.
Do not expose it directly to the public internet.

For the supported and deferred workflow combinations around the API probe, see
[CAPABILITY_MATRIX.md](CAPABILITY_MATRIX.md).
For the complete trust-boundary and deployment decision, see
[API_SECURITY.md](API_SECURITY.md).

The API has no built-in auth, no database, and no multi-user state
coordination. BO Forge uses bounded in-memory suggestion stages, but they are not
persistent and disappear whenever the API process restarts. Do not expose the
probe directly to the public internet.

## Install

```bash
pip install "bo-forge[api]"
```

From a development checkout:

```bash
./.venv/bin/pip install -e ".[dev]"
```

## Launch

Local-only:

```bash
bo-forge-api --root . --host 127.0.0.1 --port 8765
```

The bundled launcher runs one API process. Server-managed stages default to a
30-minute lifetime and 128 active batches:

```bash
bo-forge-api --root . \
  --stage-ttl-seconds 1800 \
  --max-staged-batches 128
```

Trusted LAN or lab server:

```bash
bo-forge-api --root . --host 0.0.0.0 --port 8765 --allow-network-access
```

Wildcard or non-loopback hosts expose the probe to the network. Use only on a
trusted LAN, VPN, or SSH tunnel. Anyone who can reach the API can read and write
campaign files under the configured root directory through the exposed campaign
operations. `--allow-network-access` is required for these binds, but it only
acknowledges the exposure; it does not add authentication, TLS, or authorization.

To require the preferred server-managed append workflow and remove interactive
documentation routes:

```bash
bo-forge-api --root . --server-stages-only --no-docs
```

`--server-stages-only` leaves `/campaign/suggestions/append` defined for API
compatibility but returns `403 client_bundle_append_disabled`; server-managed
`/campaign/stages/{stage_id}/append` remains available. `--no-docs` disables
`/docs`, `/redoc`, and `/openapi.json`. Neither option provides authentication.
The startup banner confirms whether client-carried append and interactive docs
are enabled. A malformed compatibility-append request still receives the normal
structured `422 request_validation` response before route handling.

SSH tunnel:

```bash
bo-forge-api --root . --host 127.0.0.1 --port 8765
ssh -L 8765:127.0.0.1:8765 user@server
```

Then open `http://127.0.0.1:8765/docs` on the client machine.

## Root-Bound Paths

The probe uses a root-bound path model.

The launcher requires an existing root directory:

```bash
bo-forge-api --root /path/to/campaign-workspace
```

Requests must use relative `config_path` and `log_path` values. Absolute paths
and paths that escape the root after symlink resolution are rejected with
structured JSON errors and do not touch files.

Example campaign reference:

```json
{
  "config_path": "configs/01_simple_2d_maximise_logei.yaml",
  "log_path": "examples/01_simple_2d_maximise_logei_campaign_log.csv"
}
```

## Endpoints

Health:

```bash
curl http://127.0.0.1:8765/health
```

Validate a campaign:

```bash
curl -X POST http://127.0.0.1:8765/campaign/validation \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv"}'
```

Summarise a campaign:

```bash
curl -X POST http://127.0.0.1:8765/campaign/summary \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv"}'
```

For fidelity campaigns, the additive summary response includes both
`fidelity_summary` and `fidelity_coverage` table payloads. Non-fidelity
campaigns return empty payloads for those keys.

Generate dry-run suggestions without mutating the CSV:

```bash
curl -X POST http://127.0.0.1:8765/campaign/suggestions/dry-run \
  -H "Content-Type: application/json" \
  -d '{"config_path":"configs/01_simple_2d_maximise_logei.yaml","log_path":"examples/01_simple_2d_maximise_logei_campaign_log.csv","batch_size":1}'
```

Contextual campaigns pass context values in the same dry-run request:

```json
{
  "config_path": "configs/16_contextual_logei.yaml",
  "log_path": "examples/16_contextual_logei_campaign_log.csv",
  "batch_size": 1,
  "context_values": {
    "feedstock_acidity": 0.25
  }
}
```

The dry-run response includes both the compatibility staged bundle and
preferred server-managed stage metadata:

```json
{
  "stage": {
    "stage_id": "opaque-server-id",
    "status": "active",
    "created_at": "...",
    "expires_at": "..."
  },
  "staged_bundle": {
    "...": "exact dry-run staged_bundle"
  }
}
```

The preferred workflow keeps the exact batch in the API process:

```text
GET    /campaign/stages
GET    /campaign/stages/{stage_id}
POST   /campaign/stages/{stage_id}/renew
POST   /campaign/stages/{stage_id}/append
DELETE /campaign/stages/{stage_id}
```

`GET /campaign/stages/{stage_id}` recovers stage metadata, suggestions, and
quality. `POST .../append`
atomically claims and appends the server-held batch, so concurrent attempts can
produce only one successful write. `DELETE` discards an active batch. Stages
expire lazily and terminal states return structured `stage_*` errors. A full
store rejects a dry-run before BO suggestion generation begins. An append claim
is not expired while its request is running because releasing it could permit a
duplicate write; a stuck in-flight claim remains visible in `/health` until the
request returns or the process restarts. `/health` reports active/appending
and in-progress reservation counts plus configured limits.

### List Stage Lifecycles

`GET /campaign/stages` returns metadata only. It never returns suggestion rows,
quality tables, fingerprints, staged bundles, or context values.

```bash
curl "http://127.0.0.1:8765/campaign/stages?include_terminal=true&status=active&status=stale&limit=50"
```

Query parameters:

- `include_terminal=false` omits consumed, discarded, stale, and expired
  tombstones by default;
- repeatable `status=` filters accept `active`, `appending`, `consumed`,
  `discarded`, `stale`, and `expired`;
- `limit=50` accepts values from 1 through 200.

Results are ordered by newest lifecycle transition, then stage ID. Listing
lazily expires old active stages and marks active stages stale when their config
or log fingerprint no longer matches. A summary includes only lifecycle times,
remaining TTL, suggestion count, root-relative config/log paths, structured
stage selection, configured context variable names, renewal count, and a concise
status reason. Defaults and safely inferred single-stage selections are reported
after backend resolution. Terminal tombstones retain only this metadata, so completed stages do
not keep DataFrames, bundles, fingerprints, or context values in memory.

### Renew An Active Stage

Renewal is explicit:

```bash
curl -X POST http://127.0.0.1:8765/campaign/stages/STAGE_ID/renew
```

Renewal works only while a stage is active and its config/log fingerprints are
still valid. It resets expiry to the configured TTL from the renewal time,
increments `renewal_count`, and preserves the original suggestions, quality,
creation time, fingerprints, context, and structured-stage selection. Reads do
not extend TTL. Expired or terminal stages cannot be revived; a claimed append
returns `stage_in_use`, and changed campaign files make the stage `stale`.

### Health Diagnostics

The additive `staging` object in `/health` reports active, appending, reserved,
and remaining capacity; oldest active/appending ages; retained terminal
tombstone counts by status; and process-local totals for created, claimed, restored, renewed,
consumed, discarded, stale, expired, and capacity-rejected stages. Health is
cheap and does not hash campaign files. Lifecycle totals reset whenever the API
process restarts and are operational diagnostics, not a persistent audit log.
The additive `deployment` object reports that authentication is absent, stage
storage is process memory, multi-worker staging is unsupported, and whether
client-carried bundles and interactive docs are enabled. Health remains cheap
and does not inspect campaign files or expose secrets.

Suggestion generation is bound to the config and log snapshots loaded before
optimization. If either file changes during optimization, the dry-run fails and
does not create a stage. Append failures restore a stage to `active` only when
both files still have the staged fingerprints. If the log changed before the
failure was returned, the stage becomes `stale` because retry safety cannot be
proven.

Stage IDs are opaque but are not authentication credentials. Anyone who can
reach the API remains within the probe's no-auth
trust boundary. Stages are process-local, are not shared across Uvicorn workers,
and disappear on restart.

## Client-Carried Compatibility Append

Send that exact bundle to `/campaign/suggestions/append` to append through the
existing `CampaignAppService.append_staged()` path:

```json
{
  "config_path": "configs/01_simple_2d_maximise_logei.yaml",
  "log_path": "examples/01_simple_2d_maximise_logei_campaign_log.csv",
  "staged_bundle": {
    "...": "exact dry-run staged_bundle"
  }
}
```

Clients should not edit the staged bundle. Append rechecks the staged bundle's
embedded `config_fingerprint` and `log_fingerprint`; append does not use a
separate `expected_log_fingerprint`. Contextual dry-runs also record
the supplied `context_values` in the staged bundle so trusted clients can retain
the context used to generate the suggestions.

Client-carried staged bundles are fingerprint and integrity checked. They are
not authenticated or signed, and a trusted client can craft a schema-valid
bundle. This compatibility path remains operational through v2.5.x, but
server-managed staging is preferred because the append payload is retained by
the API process. A successful compatibility append consumes the matching
server-held stage and marks other stages from the replaced log snapshot stale,
so compatibility clients do not leak stage capacity. Signed bundles remain
deferred.
Deployments started with `--server-stages-only` disable this compatibility
append path and return `403 client_bundle_append_disabled` without touching the
CSV. Dry-run still returns the compatibility bundle so default response shapes
remain additive and stable.

Review and observation mutations require `expected_log_fingerprint`. If the log
changed since the caller last read it, or if the fingerprint is missing, the
mutation fails without writing. Use the `log_fingerprint` returned by
validation, summary, dry-run, append, review, or observation responses before
the next mutation.

Fingerprint checks run inside the same cross-process lock as CSV validation,
mutation, atomic replacement, and post-write validation. Append, review, and
observation therefore reject stale callers rather than overwriting a newer
same-machine write. A busy lock returns `log_busy`. Locks do not coordinate
different hosts writing the same network filesystem. If post-write validation
fails, BO Forge restores the prior CSV bytes before reporting failure; if that
rollback cannot be confirmed, the error states that campaign file state is
uncertain.

Server-stage lifecycle errors use these codes:

| Code | HTTP status | Meaning |
| --- | ---: | --- |
| `stage_not_found` | 404 | The stage is unknown, including after restart or tombstone eviction. |
| `stage_expired` | 410 | The stage exceeded its configured lifetime. |
| `stage_consumed` | 409 | The batch was already appended. |
| `stage_discarded` | 409 | The batch was explicitly discarded. |
| `stage_stale` | 409 | Config/log fingerprints changed before or during append, so retry is unsafe. |
| `stage_in_use` | 409 | Another request currently owns the append claim. |
| `stage_capacity` | 503 | The process-local active-stage limit is full. |
| `log_busy` | 409 | Another local process held the campaign log lock too long. |
| `client_bundle_append_disabled` | 403 | This deployment requires server-managed stage append. |

Errors retain the existing `code`, `message`, HTTP status, and request
validation `details`, and add recovery fields:

```json
{
  "ok": false,
  "error": {
    "code": "stage_stale",
    "message": "...",
    "retryable": false,
    "suggested_action": "Refresh campaign state and generate a new dry-run."
  }
}
```

`stage_in_use`, `stage_capacity`, and `log_busy` can be retried after waiting or
freeing capacity. Expired, consumed, discarded, stale, or unknown stage IDs
cannot be retried with the same stage. `stale_log` requires refreshing campaign
state and resubmitting with the new fingerprint. Path, request-validation, and
general BO errors require correcting the request or campaign first.

Multi-objective observation requests use coupled objective values:

```json
{
  "config_path": "configs/10_multi_objective_mixed_constrained_qlogehvi.yaml",
  "log_path": "examples/10_multi_objective_mixed_constrained_campaign_log.csv",
  "row_id": "suggested-row-id",
  "expected_log_fingerprint": "current-log-fingerprint",
  "objective_values": {
    "yield_score": 70.0,
    "waste_score": 15.0
  }
}
```

Partial multi-objective values fail through the existing backend validation and
leave the CSV unchanged.

## Table Payloads

DataFrames are serialized as:

```json
{
  "columns": ["row_id"],
  "records": [{"row_id": "x"}]
}
```

Missing pandas values and NaN values become `null`; other non-JSON-native values
are converted to strings.

## Scope

The API probe does not add BO behavior, schemas, auth, CORS broadening, report
or plot endpoints, a database, persistent jobs, multi-worker stage sharing,
remote Streamlit mode, or production deployment infrastructure.

Streamlit remains the recommended local UI.
