
# OpenTimestamps Provenance Records

Project: mlops-cicd-pipeline

## Timestamped milestones

### Initial project milestone
- Git commit: a5854fe205451a9cc7c29def66633f6efa13fa32
- Proof: mlops-initial-commit.ots
- Target: README.md

### Subsequent milestone
- Git commit: f7720915acfa1aff6a2d40ef28a26c544a3d1104
- Proof: proofs/proof-commit-f7720915.ots
- Target: README.md

## Verification

Proofs are generated using OpenTimestamps Client 0.8.0.

Verification command:

```bash
ots info timestamps/mlops-initial-commit.ots
ots info timestamps/proofs/proof-commit-f7720915.ots
```

Note: Calendar attestations and Bitcoin anchoring must be independently verified before claiming completed Bitcoin verification.

