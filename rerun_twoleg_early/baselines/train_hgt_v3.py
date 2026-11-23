"""v3: train HGT (Liu et al., ESWA 2026) in the shared environment; settings in hgt_adapter (100 iterations x 10 episodes)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import drl_v3_common as C
import hgt_adapter as S
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--iterations", type=int, default=S.ITERATIONS); ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=str(C.H / "drl_v3" / "hgt")); ap.add_argument("--valid-every", type=int, default=10)
    a = ap.parse_args()
    C.train_parallel("hgt_adapter", name="hgt", out=a.out, iterations=a.iterations, episodes_per_iter=S.EPISODES, ppo=S.PPO,
                     workers=a.workers, valid_every=a.valid_every, lr_schedule=S.lr_schedule)
