"""v3 section 2: run any pipeline module with per-seed random events (training, validation and test instances).
generator.generate_instance is wrapped BEFORE the target module is imported, so every `from ... import generate_instance`
binds to the wrapped version. Events are drawn by event_blind.randomize_events(instance, seed, level), the same draw as
the hidden-event test sets. Usage: python v3_random_events.py <module> [args...]"""
import runpy
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
for p in (H / "src", H / "baselines", H.parent / "twoleg"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import su_modfjsp_ha_pc.generator as G

_orig = G.generate_instance


def generate_instance(level, size, seed):
    from event_blind import randomize_events
    return randomize_events(_orig(level, size, seed), seed, level)


generate_instance.v3_random_events = True
G.generate_instance = generate_instance
G.generate_instance_fixed_script = _orig

if __name__ == "__main__":
    target = sys.argv[1]
    sys.argv = [target] + sys.argv[2:]
    runpy.run_module(target, run_name="__main__", alter_sys=True)
