# Streamlit Deployment And Safety Guide

BO Forge v1.3.1 documents local and trusted-network use of the existing
Streamlit workbench. This guide covers deployment choices only. It does not
change BO behavior, YAML/CSV semantics, launcher behavior, authentication,
storage, or app workflow logic.

The experimental FastAPI probe has separate guidance in
[API_PROBE.md](API_PROBE.md). The Streamlit workbench remains the recommended
local UI.

## Safety Model

BO Forge is a local-first workbench:

- BO Forge v1.3.1 has no built-in auth.
- It has no multi-user state coordination.
- It has no database or server-side campaign store.
- It is not hardened for direct public internet exposure.
- There is no safe unauthenticated public internet exposure mode.
- Anyone who can access the app can interact with host filesystem paths selected
  in the app.
- Treat app access as host filesystem access within the selected working paths.
- CSV campaign logs remain the source of truth.

Use these operating rules for any shared or remote session:

- Use a dedicated campaign working directory.
- Work on copied CSV logs, not seed example logs.
- Back up CSV logs before shared or remote sessions.
- Avoid simultaneous writes from multiple users.
- Prefer VPN, SSH tunnel, or reverse proxy auth for remote access.
- Do not expose an unauthenticated BO Forge app directly to the public internet.

## Mode 1: Local-Only Workstation

Use this when the app and browser run on the same machine:

```bash
bo-forge-app
bo-forge-app --host 127.0.0.1 --port 8501
```

Open:

```text
http://127.0.0.1:8501
```

This is the safest default. The app binds to loopback only, so other machines on
the network cannot connect directly.

## Mode 2: Trusted LAN Or Lab Server

Use this only on a trusted LAN, lab network, VPN-backed subnet, or similarly
controlled environment:

```bash
bo-forge-app --host 0.0.0.0 --port 8501 --no-browser
```

Open from another trusted device:

```text
http://<host-machine-lan-ip>:8501
```

Wildcard or non-loopback hosts expose the app to the network and trigger the
same launcher warning. Examples include:

- `0.0.0.0`
- `::`
- a LAN IP such as `192.168.1.25`
- a LAN hostname such as `lab-workstation.local`

Use a dedicated campaign working directory and copied CSV logs. Avoid
simultaneous writes from multiple users, because BO Forge does not coordinate
multi-user state or file locks at the app layer.

## Mode 3: SSH Tunnel Or VPN

Prefer this for remote access when users should not bind the app to a network
interface:

```bash
bo-forge-app --host 127.0.0.1 --port 8501 --no-browser
ssh -L 8501:127.0.0.1:8501 user@server
```

Open in the client machine's browser:

```text
http://127.0.0.1:8501
```

The Streamlit app remains bound to loopback on the server. Access is mediated by
the SSH tunnel or VPN rather than by exposing the app directly to the network.

## Mode 4: Reverse Proxy With External Auth

Reverse proxy deployment is advanced and user-managed. BO Forge does not ship a
proxy config, auth system, service unit, Dockerfile, or public deployment
recipe.

If you place BO Forge behind a reverse proxy:

- require external authentication before any app exposure;
- terminate TLS and access control outside BO Forge;
- keep the Streamlit process bound to a private interface when possible;
- do not publish an unauthenticated public proxy route;
- document who can access the app and what host-local files it can reach;
- keep CSV log backups before shared sessions.

BO Forge intentionally does not provide a copy-paste unauthenticated public proxy
configuration.

## Process Manager Sketch

If you use a process manager, keep it user-managed and record:

- working directory: the dedicated campaign directory or repository checkout;
- environment: the Python environment with `bo-forge[app]` installed;
- command: for example `bo-forge-app --host 127.0.0.1 --port 8501 --no-browser`;
- logs: stdout/stderr capture path and rotation policy;
- restart policy: when to restart, and who owns stopping the app;
- backups: CSV log backup location and cadence.

Do not use a process manager to turn BO Forge into a public unauthenticated web
service.
