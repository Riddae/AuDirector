# AuDirector: A Self-Reflective Closed-Loop Framework for Immersive Audio Storytelling

## Abstract
Despite advances in text and visual generation, creating coherent long-form audio narratives remains challenging. Existing frameworks often exhibit limitations such as mismatched character settings with voice performance, insufficient self-correction mechanisms, and limited human interactivity.

To address these challenges, we propose **AuDirector**, a self-reflective closed-loop multi-agent framework. Specifically, it involves an Identity-Aware Pre-production mechanism that transforms narrative texts into character profiles and utterance-level emotional instructions to retrieve suitable voice candidates and guide expressive speech synthesis, thereby promoting context-aligned voice adaptation.

To enhance quality, a Collaborative Synthesis and Correction module introduces a closed-loop self-correction mechanism to systematically audit and regenerate defective audio components. Furthermore, a Human-Guided Interactive Refinement module facilitates user control by interpreting natural language feedback to interactively refine the underlying scripts.

Experiments demonstrate that AuDirector achieves superior performance compared to state-of-the-art baselines in structural coherence, emotional expressiveness, and acoustic fidelity.
## Framework Architecture

![AuDirector Framework](demo/AuDirector.png)

## Key Features

- **Identity-Aware Pre-Production**: Transforms narrative texts into character profiles and emotional instructions for voice adaptation
- **Collaborative Synthesis and Correction**: Closed-loop self-correction mechanism for quality assurance
- **Human-Guided Interactive Refinement**: Natural language feedback interpretation for interactive script refinement
- **Multi-Agent Architecture**: Orchestrates the complete audiobook generation process
- **High-Quality Audio Generation**: Supports TTS, voice matching, BGM, and SFX generation

## Installation

### Prerequisites
- Python 3.8+
- Access to required API endpoints (configured in `config.json`)

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd AuDirector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API endpoints in `config.json`:
```json
{
  "api": {
    "tts": "your-tts-endpoint",
    "speaker_matching": "your-speaker-matching-endpoint",
    "music_gen": "your-music-gen-endpoint"
  },
  "llm": {
    "api_key": "your-llm-api-key",
    "base_url": "your-llm-base-url",
    "model_name": "your-model-name"
  }
}
```

## Usage

### Command Line Interface
Run the complete workflow with a story description:

```bash
python run.py "Your story description here" --output projects --name my_audiobook
```

### Parameters
- `input`: The story description or prompt for the audiobook (required)
- `--output`, `-o`: Root directory for all project outputs (default: "projects")
- `--name`, `-n`: Project name prefix (default: "my_audiobook")

### Python API
```python
from workflow import AudioBookWorkflow

# Initialize workflow
workflow = AudioBookWorkflow(output_base_dir="output_directory")

# Run complete workflow
result = workflow.run_complete_workflow("Your story description")

if result.success:
    print(f"Audio generated successfully: {result.output_files['final_mix']}")
else:
    print(f"Generation failed: {result.message}")
```

## Workflow Steps

1. **Content Generation**: Transform user input into structured dialogue format
2. **Speaker Matching**: Match characters to appropriate voice profiles
3. **Dialogue Generation**: Generate detailed dialogue with emotional vectors
4. **Speech Synthesis**: Generate speech audio with timestamps
5. **Script Generation**: Create audio production script
6. **Effects Generation**: Generate BGM and SFX
7. **Final Mixing**: Combine all audio tracks into final output

## Configuration

### API Endpoints
Configure the following services in `config.json`:
- **TTS**: Text-to-speech synthesis
- **Speaker Matching**: Character-to-voice matching
- **Music Generation**: Background music generation
- **CLAP**: Audio classification and processing
- **MIMO Audio**: Multi-modal audio processing

### Audio Settings
- Supported formats: WAV, MP3, FLAC
- Default gap between dialogue: 1500ms
- Fade in/out settings for smooth transitions
- Quality thresholds for automatic regeneration

### LLM Configuration
- Supports various LLM providers
- Configurable temperature and token limits
- Timeout settings for API calls

## Citation



## License

[Add license information here]
