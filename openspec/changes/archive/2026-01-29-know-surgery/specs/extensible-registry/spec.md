## ADDED Requirements

### Requirement: Plugin metadata registry
The system SHALL associate metadata with registered models, methods, and datasets, including task type, modality, and resource requirements.

#### Scenario: Read plugin metadata
- **WHEN** a component is registered
- **THEN** its metadata is available to the pipeline and future GUI for validation and selection

### Requirement: Compatibility validation
The system SHALL validate model-method-dataset compatibility before execution and SHALL fail fast with an actionable error message for incompatible combinations.

#### Scenario: Reject incompatible combination
- **WHEN** a selected method is incompatible with the chosen model or dataset
- **THEN** the pipeline aborts before training and reports the incompatibility reason
