## ADDED Requirements

### Requirement: Multimodal data schema
The system SHALL define a multimodal sample schema that supports text and image inputs and aligns with existing dataset loading interfaces.

#### Scenario: Load multimodal sample
- **WHEN** a multimodal dataset is registered and loaded
- **THEN** the data loader provides text and image fields according to the standard schema

### Requirement: Multimodal model adapters
The system SHALL support adapter interfaces for multimodal models (e.g., LLaVA, Qwen-VL) with encode/forward/generate entrypoints.

#### Scenario: Run multimodal evaluation
- **WHEN** a multimodal model is selected with a supported evaluator
- **THEN** the system executes evaluation using the multimodal adapter interface

### Requirement: Multimodal benchmarks and metrics
The system SHALL support multimodal benchmarks and metrics for unlearning/editing (e.g., MLLMU-Bench, MMMU, LLAVA-bench, CLEAR, MMEdit, MMKE).

#### Scenario: Configure multimodal benchmark
- **WHEN** a multimodal benchmark is selected
- **THEN** the system runs the benchmark and reports results in the standard metrics format
