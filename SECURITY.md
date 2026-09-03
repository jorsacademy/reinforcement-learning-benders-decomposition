# Security policy

This repository is an offline research prototype. It does not execute model-generated code, call external AI services, or require credentials.

## Data and model files

JSON instances and NumPy checkpoints are untrusted input. The implementation:

- validates JSON structure and numeric finiteness;
- loads NumPy archives with `allow_pickle=False`;
- validates checkpoint version, action order, shape, and finite Q values;
- never evaluates arbitrary Python expressions from files.

Do not process confidential production data without a separate data-governance review. Generated reports can reveal facility costs, capacities, demand scenarios, and first-stage decisions.

## Denial-of-service considerations

Large extensive-form models, scenario counts, or first-stage enumeration can consume substantial time and memory. The enumeration oracle is explicitly capped. Production deployments should add external resource limits, input-size limits, and solver time limits.

## Reporting a vulnerability

Report security concerns privately to the repository owner before public disclosure. Include a minimal reproducer, affected version or commit, and expected impact.
