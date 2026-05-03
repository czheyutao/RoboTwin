#!/usr/bin/env python3
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from datetime import datetime
from typing import Dict, List, Sequence, Tuple


TASKS: Sequence[str] = (
    "move_stapler_pad",
    "place_fan",
    "handover_mic",
    "open_microwave",
    "place_can_basket",
    "place_dual_shoes",
    "stack_blocks_three",
    "move_can_pot",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
)


@dataclass(frozen=True)
class Job:
    task: str
    config: str
    gpu: int

    @property
    def name(self) -> str:
        return f"{self.task}:{self.config}:gpu{self.gpu}"


def parse_gpu_list(raw: str) -> List[int]:
    gpus: List[int] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Invalid GPU id: {p}")
        gpus.append(int(p))
    if not gpus:
        raise ValueError("GPU list is empty")
    return gpus


def build_jobs(
    tasks: Sequence[str],
    clean_cfg: str,
    rand_cfg: str,
    gpus: Sequence[int],
) -> List[Job]:
    job_pairs: List[Tuple[str, str]] = []
    for task in tasks:
        job_pairs.append((task, clean_cfg))
        job_pairs.append((task, rand_cfg))

    jobs: List[Job] = []
    for idx, (task, cfg) in enumerate(job_pairs):
        gpu = gpus[idx % len(gpus)]
        jobs.append(Job(task=task, config=cfg, gpu=gpu))
    return jobs


def run_job(job: Job, repo_root: Path, log_dir: Path, retry: int) -> Tuple[Job, bool, int]:
    log_file = log_dir / f"{job.task}_{job.config}_gpu{job.gpu}.log"
    cmd = ["bash", "collect_data.sh", job.task, job.config, str(job.gpu)]
    attempts = retry + 1

    for i in range(1, attempts + 1):
        start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_file.open("a", encoding="utf-8") as fp:
            fp.write(f"\n===== {job.name} | attempt {i}/{attempts} | start {start_ts} =====\n")
            fp.write(f"[Command] {' '.join(cmd)}\n")
            fp.flush()
            t0 = time.monotonic()
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                stdout=fp,
                stderr=subprocess.STDOUT,
                check=False,
            )
            elapsed = time.monotonic() - t0
            end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fp.write(
                f"[Exit] code={proc.returncode}, elapsed={elapsed:.1f}s, end={end_ts}\n"
            )
        if proc.returncode == 0:
            return job, True, i
    return job, False, attempts


def render_progress(done: int, total: int, ok: int, fail: int, width: int = 32) -> str:
    if total <= 0:
        return "[--------------------------------]   0.0% (0/0) ok=0 fail=0"
    ratio = done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    pct = ratio * 100
    return f"[{bar}] {pct:5.1f}% ({done}/{total}) ok={ok} fail={fail}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Multi-GPU data collection pool for RoboTwin."
    )
    parser.add_argument("--gpus", default="0,1,2,3", help="GPU list, e.g. 0,1,2,3")
    parser.add_argument(
        "--per-gpu",
        type=int,
        default=2,
        help="Concurrent slots per GPU.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Override total worker count. 0 means len(gpus) * per_gpu.",
    )
    parser.add_argument("--clean-cfg", default="demo_clean")
    parser.add_argument("--rand-cfg", default="demo_randomized")
    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help="Retry times for each failed job.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned jobs without execution.",
    )
    args = parser.parse_args()

    if args.per_gpu <= 0:
        raise ValueError("--per-gpu must be >= 1")
    if args.max_workers < 0:
        raise ValueError("--max-workers must be >= 0")
    if args.retry < 0:
        raise ValueError("--retry must be >= 0")

    gpus = parse_gpu_list(args.gpus)
    repo_root = Path(__file__).resolve().parent.parent
    log_dir = repo_root / "logs" / "collect_pool"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(
        tasks=TASKS,
        clean_cfg=args.clean_cfg,
        rand_cfg=args.rand_cfg,
        gpus=gpus,
    )
    total = len(jobs)
    workers = args.max_workers if args.max_workers > 0 else len(gpus) * args.per_gpu

    print(f"[Plan] GPUs={gpus}, per_gpu={args.per_gpu}, workers={workers}")
    print(f"[Plan] jobs={total}, clean_cfg={args.clean_cfg}, rand_cfg={args.rand_cfg}")
    print(f"[Plan] logs={log_dir}")
    for idx, job in enumerate(jobs, start=1):
        print(f"  {idx:02d}. {job.name}")

    if args.dry_run:
        return 0

    success = 0
    failed: List[Tuple[Job, int]] = []
    done = 0
    start_all = time.monotonic()

    print(f"[Progress] {render_progress(done, total, success, len(failed))}")

    jobs_by_gpu: Dict[int, List[Job]] = {gpu: [] for gpu in gpus}
    for job in jobs:
        jobs_by_gpu[job.gpu].append(job)

    all_futures = []
    pools: List[ThreadPoolExecutor] = []
    try:
        for gpu in gpus:
            pool = ThreadPoolExecutor(max_workers=args.per_gpu, thread_name_prefix=f"gpu{gpu}")
            pools.append(pool)
            for job in jobs_by_gpu[gpu]:
                fut = pool.submit(run_job, job, repo_root, log_dir, args.retry)
                all_futures.append(fut)
                log_file = log_dir / f"{job.task}_{job.config}_gpu{job.gpu}.log"
                print(f"[Submit] {job.name} -> {log_file}")

        for fut in as_completed(all_futures):
            job, ok, attempts = fut.result()
            done += 1
            log_file = log_dir / f"{job.task}_{job.config}_gpu{job.gpu}.log"
            if ok:
                success += 1
                print(f"[{done}/{total}] OK    {job.name} (attempts={attempts}) log={log_file}")
            else:
                failed.append((job, attempts))
                print(f"[{done}/{total}] FAIL  {job.name} (attempts={attempts}) log={log_file}")
            print(f"[Progress] {render_progress(done, total, success, len(failed))}")
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=False)

    elapsed_all = time.monotonic() - start_all
    print(
        f"[Summary] success={success}, failed={len(failed)}, total={total}, elapsed={elapsed_all:.1f}s"
    )
    if failed:
        print("[Failed Jobs]")
        for job, attempts in failed:
            print(f"  - {job.name} (attempts={attempts})")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(2)
