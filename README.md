# Video2Prompt

Turn videos into actionable prompts for your AI tools.

Video2Prompt transcribes a video, analyzes what is happening on screen, and produces a structured prompt you can paste into Claude, ChatGPT, Cursor, or any coding agent. It supports two workflows:

| Mode | Command | Output |
| --- | --- | --- |
| **Feedback** | `analyze` | Implementation prompt for your coding agent |
| **Extract** | `extract` | Playbook generation prompt for content recreation |

## How it works

Both modes follow the same pipeline:

1. **Transcribe** — local Whisper or cloud transcription
2. **Detect segments** — feedback moments or content beats
3. **Extract frames** — screenshots at key timestamps
4. **Vision analysis** — what is on screen vs what is being said
5. **Synthesize** — final structured output + markdown prompt

Work is cached under `work/` so interrupted runs resume automatically.

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` and `ffprobe` on PATH)
- An LLM provider (see below)

## Setup

```bash
git clone <your-repo-url>
cd video2prompt

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env
```

Edit `.env` with your provider credentials. On first run, `input/`, `output/`, and `work/` are created automatically.

Drop your video files into `input/` and run a command below.

## Usage

### Interactive (recommended)

```bash
video2prompt start
```

The wizard walks you through:

1. **Mode** — feedback or content extraction
2. **Provider** — Cursor, Claude CLI, Codex CLI, OpenAI, or auto
3. **Models** — vision and synthesis models
4. **Transcription** — local, online, or auto
5. **Video** — pick from `input/` or enter a path

After each job finishes, you are asked whether to process another video. Settings are kept — only the next video is prompted.

### Client feedback → implementation prompt

For screen recordings where a client gives product feedback:

```bash
video2prompt analyze input/your-video.mp4
```

With codebase context (maps feedback to files and symbols):

```bash
video2prompt analyze input/your-video.mp4 --project /path/to/repo
```

**Outputs:**

- `output/<video>-implementation-prompt.md` — paste into your coding agent
- `output/<video>-feedback.json` — structured analysis

### Content extraction → playbook prompt

For any video where you want to understand and recreate the content style:

```bash
video2prompt extract input/your-video.mp4
```

**Outputs:**

- `output/<video>-playbook-prompt.md` — paste into Claude or ChatGPT to generate a reproducible content playbook
- `output/<video>-content-features.json` — structured feature extraction

**What gets extracted:**

- Hooks and opening techniques
- Scripting structure and pacing flow
- Phrasing and language patterns
- What is shown on screen vs what is said
- Speaking and delivery style
- Segment-by-segment breakdown

## Providers

Set `LLM_PROVIDER` in `.env` or pass `--provider`:

| Provider | Auth | Best for |
| --- | --- | --- |
| `cursor` | `CURSOR_API_KEY` | Cursor API / subscription |
| `claude` | `claude login` | Claude Code CLI subscription |
| `codex` | `codex login` | Codex CLI / ChatGPT subscription |
| `openai` | `OPENAI_API_KEY` | Pay-as-you-go API |
| `subscription` | auto | Tries Claude CLI, then Codex CLI |

Transcription backends: `local` (faster-whisper), `online` (OpenAI or Codex), or `auto`.

## Project layout

```
video2prompt/
├── input/          # Your videos (gitignored, auto-created)
├── output/         # Generated prompts and JSON (gitignored, auto-created)
├── work/           # Cached transcripts, frames, intermediate analysis (gitignored)
├── prompts/        # LLM prompt templates
├── src/            # Source code
└── tests/
```

Override default paths in `.env`:

```
INPUT_DIR=input
OUTPUT_DIR=output
WORK_DIR=work
```

## Common flags

```bash
video2prompt analyze --help
video2prompt extract --help
```

| Flag | Description |
| --- | --- |
| `--provider` | LLM provider override |
| `--model` | Reasoning / detection model |
| `--vision-model` | Model for screenshot analysis |
| `--synthesis-model` | Model for final prompt synthesis |
| `--transcription` | `auto`, `local`, or `online` |
| `--force` | Re-run all stages (ignore cache) |
| `--no-keep-work` | Delete intermediate files after completion |
| `--verbose` | Show detailed logs |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
