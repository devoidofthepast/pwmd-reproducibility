"""v3: train DANIEL (Wang et al.) in the shared environment; settings in daniel_adapter.PPO, 1000 updates x 20 episodes."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drl_v3_common as C
import daniel_adapter as S
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--iterations", type=int, default=1000); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(C.H / "drl_v3" / "daniel")); ap.add_argument("--valid-every", type=int, default=50)
    a = ap.parse_args()
    C.train_parallel("daniel_adapter", name="daniel", out=a.out, iterations=a.iterations, episodes_per_iter=20, ppo=S.PPO,
                     workers=a.workers, valid_every=a.valid_every)
