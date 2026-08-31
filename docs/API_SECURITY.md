# Experimental API Security And Trust Boundaries

BO Forge v3.0.2 includes deployment safeguards for deliberate local and
trusted-network use of the experimental FastAPI probe. These safeguards do not
make the probe a production service. The API has no built-in authentication,
authorization, user identity, TLS termination, or persistent audit log.

## Assets And Trust Boundary

The protected assets are campaign YAML files, CSV logs, staged suggestions,
host compute time, and any other files reachable under the configured API
`--root`. Anyone who can call the API can exercise the exposed campaign reads,
suggestion computation, and mutation operations within that root.

The root-bound path guard prevents request paths and resolved symlinks from
escaping `--root`. It does not distinguish users or authorize one caller over
another. Treat API access as permission to read and mutate campaign files under
the configured root.

## Existing Safeguards

BO Forge currently provides:

- root-bound relative config and log paths, including symlink resolution;
- config/log fingerprints for stale-state and staged-bundle integrity checks;
- server-managed staged batches with opaque random IDs, TTL, capacity limits,
  lifecycle state, and exactly-once append claims;
- same-machine file locks around append, review, and observation mutations;
- mode-preserving temporary-file replacement, post-write validation, and
  rollback to the prior CSV bytes when validation fails;
- structured errors and recovery guidance;
- no permissive wildcard CORS configuration;
- launcher warnings and required acknowledgement for non-loopback binds.

These controls reduce accidental path escape, stale writes, duplicate stage
append, and same-machine write races. They do not authenticate callers or stop
an authorized network client from requesting expensive suggestion work.

## Deployment Modes

Preferred local-only launch:

```bash
bo-forge-api --root /path/to/campaigns --host 127.0.0.1 --port 8765
```

For remote use, prefer an SSH tunnel or VPN while keeping the API on loopback.
If a trusted LAN bind is necessary, acknowledgement is explicit:

```bash
bo-forge-api --root /path/to/campaigns \
  --host 0.0.0.0 --port 8765 --allow-network-access
```

`--allow-network-access` is acknowledgement only. It does not add security.
Do not expose this unauthenticated listener directly to the public internet.

An advanced deployment may place the API behind an externally authenticated
TLS reverse proxy. Proxy configuration, identity, authorization, certificates,
rate limiting, and audit logging remain user-managed. Keep the API on a private
interface and restrict its root to a dedicated campaign working directory.

## Stage And Bundle Limits

Server-managed stages are the preferred integrity workflow because suggestions
remain in process memory and append claims the exact held batch. Stage IDs are
opaque but are not credentials. They can be used by any caller who obtains
them. Stages disappear on process restart and are not shared across workers.
The bundled launcher therefore runs one API process.

Client-carried bundles remain a trusted-client compatibility path through
v2.5.x. Fingerprints detect modification relative to campaign snapshots, but
bundles are neither authenticated nor signed. A trusted client can construct a
schema-valid payload. Deployments can disable that append route:

```bash
bo-forge-api --root /path/to/campaigns --server-stages-only
```

Interactive docs can also be disabled with `--no-docs`. This removes `/docs`,
`/redoc`, and `/openapi.json`; it is not an access-control mechanism.

## Operational Risks

- Unauthorized callers can mutate campaign logs and consume compute.
- Multiple Uvicorn workers do not share server-managed stages.
- Stage state and lifecycle counters are process-local and disappear on restart.
- Same-machine locks do not coordinate separate hosts writing one shared file.
- Capacity limits bound retained stages but are not full request rate limits.
- Disabling docs reduces discovery surfaces but does not secure endpoints.

Use copied campaign logs, maintain backups, restrict the working root, avoid
multi-host writes, and monitor the process and reverse proxy when shared access
is enabled.

## Deferred Production Requirements

Built-in bearer authentication is deferred because credentials without TLS,
identity management, authorization, rotation, and auditing would create a
misleading security boundary. Signed client bundles are also deferred: signing
could establish origin or integrity but would not authorize the caller, and
server-managed stages already provide the preferred payload-integrity path.

Any future production API would require an explicit threat model, authenticated
identity, authorization policy, TLS, credential lifecycle, rate limits,
persistent multi-worker state, audit logging, deployment guidance, and security
review. None of those guarantees are claimed by this experimental probe.
