## ADDED Requirements

### Requirement: Standard evaluation artifacts
The system SHALL emit standardized evaluation artifacts for each run, including a manifest, aggregated metrics, and optional sample traces.

#### Scenario: Generate manifest
- **WHEN** a run completes evaluation
- **THEN** the system writes a manifest containing config snapshot, model identifiers, seed, and runtime environment info

#### Scenario: Generate metrics
- **WHEN** evaluation finishes
- **THEN** the system writes a metrics file with per-metric results and aggregated scores

### Requirement: Graceful degradation without retain logs
The system SHALL allow evaluation to proceed when retain logs are missing and SHALL clearly mark missing dependent metrics.

#### Scenario: Missing retain logs
- **WHEN** retain logs are not provided for retain-dependent metrics
- **THEN** the system skips those metrics and records them as missing in outputs
