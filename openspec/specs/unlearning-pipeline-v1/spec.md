# unlearning-pipeline-v1

    ## Purpose
    This specification defines requirements for the unlearning-pipeline-v1 capability.

    ## Requirements

    ## ADDED Requirements

### Requirement: One-click unlearning pipeline runner
The system SHALL provide a Python pipeline runner that orchestrates unlearning followed by evaluation using the existing train.py and eval.py entrypoints and Hydra configs.

#### Scenario: Run pipeline with a config
- **WHEN** a user runs the pipeline with a valid experiment config path
- **THEN** the system executes unlearning first and evaluation second using the same config schema and produces outputs under the configured saves directory

#### Scenario: Missing data without auto-download
- **WHEN** required datasets are missing
- **THEN** the system reports the missing data and does not attempt to auto-download in v1

### Requirement: TOFU demo configs and flexible splits
The system SHALL include two default TOFU demo configurations for Llama-3.2-1B Instruct and Qwen3-1.7B Base, and SHALL allow overriding TOFU splits via config or CLI overrides.

#### Scenario: Use default demo configuration
- **WHEN** a user runs a demo config without overrides
- **THEN** the system uses the default model and TOFU split values defined in that config

#### Scenario: Override TOFU splits
- **WHEN** a user provides override values for forget/retain/holdout splits
- **THEN** the pipeline uses the overridden split values in dataset loading and evaluation
