# Security Policy

## Supported versions

Security fixes are evaluated for the current release line. Older releases may
receive a fix when impact and compatibility make that practical, but BO Forge
does not currently promise long-term support for multiple release lines.

## Reporting a vulnerability

Do not disclose a sensitive vulnerability in a public issue, discussion, pull
request, or campaign log.

Use GitHub's private vulnerability reporting flow when the repository's
**Security** tab offers **Report a vulnerability**. This file does not enable or
verify that repository setting. If the private form is unavailable, contact the
maintainer through the GitHub profile linked by the repository and ask to
establish a private reporting channel without including vulnerability details
in the public message.

Include:

- affected BO Forge version and commit, when known;
- operating system and Python version;
- affected interface (core, CLI, Streamlit, or experimental API);
- reproduction steps or a minimal proof of concept;
- expected and observed impact;
- whether campaign files, credentials, or network access are involved;
- suggested mitigations, if available.

Allow maintainers time to confirm impact and coordinate a fix before public
disclosure.

## Product security boundary

This policy covers private reporting for BO Forge itself. It is distinct from
[`docs/API_SECURITY.md`](docs/API_SECURITY.md), which documents the current
deployment and trust model of the experimental FastAPI adapter.

The API adapter has no built-in authentication or authorization and is intended
only for localhost or explicitly trusted LAN/VPN/SSH-tunnel environments. It is
not a production multi-user backend. A report that merely restates those
documented limitations is not a newly discovered vulnerability; a bypass of a
documented path, staging, fingerprint, locking, or deployment safeguard may be.
