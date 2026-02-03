## ADDED Requirements

### Requirement: PEFT-based knowledge injection
The system SHALL support knowledge injection via PEFT methods including LoRA, DoRA, AdaLoRA, LoReFT, and BREP-REFT.

#### Scenario: Run LoRA-based injection
- **WHEN** a user selects LoRA for injection and provides a dataset
- **THEN** the system performs the injection and reports outcomes using the shared evaluation framework

### Requirement: Injection evaluation hooks
The system SHALL define evaluation hooks for injection tasks, including multimodal metrics where applicable.

#### Scenario: Evaluate injection outcomes
- **WHEN** injection evaluation runs
- **THEN** the system reports injection metrics defined for the chosen datasets and modalities
