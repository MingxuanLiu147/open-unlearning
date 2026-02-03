## ADDED Requirements

### Requirement: GUI starts at end of v2 and shares config schema
The system SHALL implement a Gradio-based GUI starting at the end of v2, and SHALL use the same config schema as the CLI for import/export compatibility.

#### Scenario: Import CLI config
- **WHEN** a user uploads a CLI config file in the GUI
- **THEN** the GUI loads the configuration and prepares a runnable setup

### Requirement: UltraRAG-style wizard flow (progressive)
The GUI SHOULD provide a wizard-style flow for selecting model, method, dataset, and evaluation, and SHALL allow a one-click run once configured.

#### Scenario: Wizard-based run
- **WHEN** a user completes the wizard selections
- **THEN** the GUI starts the pipeline run and displays progress

### Requirement: Result visualization (progressive)
The GUI SHOULD provide basic visualization of run status and key metrics, with expansion over time.

#### Scenario: Show run summary
- **WHEN** a pipeline run completes
- **THEN** the GUI displays a summary of metrics and links to output artifacts
