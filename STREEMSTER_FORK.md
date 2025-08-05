# Streemster Fork of Wan Video 2.1

This is a modified version of Wan Video 2.1 with added progress tracking and cancellation support.

## Quick Reference

### Added Parameters

Both `WanT2V.generate()` and `WanI2V.generate()` now accept:

```python
progress_callback: Optional[Callable[[int, int, Optional[str]], None]] = None
cancel_fn: Optional[Callable[[], bool]] = None
```

### Progress Callback

Called with `(step, total, stage)`:
- `step`: Current step number (0 to total)
- `total`: Total number of steps
- `stage`: One of "prepare", "sample", "finalize" (or None)

### Cancellation Function

Should return `True` to cancel generation. Raises `RuntimeError("Canceled")`.

### Environment Variables

- `WAN_TQDM=0/1` - Control TQDM progress bars (default: 0)
- `WAN_AGGRESSIVE_OFFLOAD=0/1` - Memory clearing frequency (default: 0)

## Example

```python
from wan.text2video import WanT2V

def on_progress(step, total, stage):
    print(f"{stage or 'Progress'}: {step}/{total}")

def should_cancel():
    return False  # Check your cancellation logic

model = WanT2V(config, checkpoint_dir)
video = model.generate(
    input_prompt="A beautiful sunset",
    progress_callback=on_progress,
    cancel_fn=should_cancel
)
```

## Changes from Original

See [/opt/streemster-ai-video/aitools/wan-standalone/FORK_CHANGES.md](../FORK_CHANGES.md) for detailed list of modifications.