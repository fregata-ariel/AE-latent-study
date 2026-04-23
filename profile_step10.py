"""Step 10 bottleneck profiling harness.

Non-invasive: monkey-patches data prep, train step, and evaluation entry
points to collect timings, without touching the source modules.

Driven by environment variables:

  PROFILE_EPOCHS    (default 2)      override config.train.num_epochs
  PROFILE_PATIENCE  (default 999)    override config.train.patience (no early stop)
  PROFILE_TRACE     (default 0)      if 1, start jax.profiler trace during training
  PROFILE_WORKDIR   (default runs/profile_step10_single)
  PROFILE_OUTDIR    (default docs/step10_profiling/raw)

Outputs to $PROFILE_OUTDIR:
  pid.txt                    PID for attaching py-spy / nvidia-smi
  macro_timings.json         train_and_evaluate vs run_full_evaluation vs other
  train_step_timings.json    per-batch CPU prep + JIT step timings
  eval_step_timings.json     per-metric timings inside run_full_evaluation
  jit_compiles.log           set via stderr redirect at launch
  jax_trace/                 (only when PROFILE_TRACE=1) Perfetto-readable dir
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from functools import wraps

import jax


PROFILE_OUTDIR = os.environ.get('PROFILE_OUTDIR', 'docs/step10_profiling/raw')
PROFILE_WORKDIR = os.environ.get('PROFILE_WORKDIR', 'runs/profile_step10_single')
PROFILE_EPOCHS = int(os.environ.get('PROFILE_EPOCHS', '2'))
PROFILE_PATIENCE = int(os.environ.get('PROFILE_PATIENCE', '999'))
PROFILE_TRACE = os.environ.get('PROFILE_TRACE', '0') == '1'

os.makedirs(PROFILE_OUTDIR, exist_ok=True)

with open(os.path.join(PROFILE_OUTDIR, 'pid.txt'), 'w') as _pid_file:
    _pid_file.write(f'{os.getpid()}\n')


class _Timer:
    """Accumulate call count + total seconds per label."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)
        self.per_call: dict[str, list[float]] = defaultdict(list)

    def record(self, label: str, seconds: float, keep_per_call: bool) -> None:
        self.totals[label] += seconds
        self.counts[label] += 1
        if keep_per_call:
            self.per_call[label].append(seconds)

    def dump(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(
                {
                    'totals_seconds': dict(self.totals),
                    'counts': dict(self.counts),
                    'per_call_seconds': {k: v for k, v in self.per_call.items()},
                },
                f,
                indent=2,
            )


macro_timer = _Timer()
train_timer = _Timer()
eval_timer = _Timer()


def _wrap(label: str, fn, timer: _Timer, *, block_arg: int | None = None, keep_per_call: bool = False):
    """Return fn wrapped with a perf_counter timer.

    If ``block_arg`` is given, ``block_until_ready()`` is called on
    ``result[block_arg]`` to force GPU sync before the timer stops — used
    when timing JIT-compiled device work.
    """

    @wraps(fn)
    def inner(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        if block_arg is not None and out is not None:
            try:
                jax.block_until_ready(out[block_arg])
            except Exception:
                pass
        t1 = time.perf_counter()
        timer.record(label, t1 - t0, keep_per_call)
        return out

    return inner


# ---------- Monkey-patch training side ---------- #

import train.trainer as trainer_mod  # noqa: E402

trainer_mod._make_lattice_partner_batch = _wrap(
    'cpu_prep.make_lattice_partner_batch',
    trainer_mod._make_lattice_partner_batch,
    train_timer,
    keep_per_call=True,
)
trainer_mod._reduce_tau_batch_to_fd_coords = _wrap(
    'cpu_prep.reduce_tau_batch_to_fd_coords',
    trainer_mod._reduce_tau_batch_to_fd_coords,
    train_timer,
    keep_per_call=True,
)
trainer_mod._make_j_rank_targets = _wrap(
    'cpu_prep.make_j_rank_targets',
    trainer_mod._make_j_rank_targets,
    train_timer,
    keep_per_call=True,
)
trainer_mod._make_teacher_quotient_batch = _wrap(
    'cpu_prep.make_teacher_quotient_batch',
    trainer_mod._make_teacher_quotient_batch,
    train_timer,
    keep_per_call=True,
)


# Wrap the factory that builds the JIT step, so the returned step is timed.
_orig_make_train_step_factorized = trainer_mod._make_train_step_factorized_lattice_vae


def _make_train_step_factorized_timed(*args, **kwargs):
    raw_step_fn = _orig_make_train_step_factorized(*args, **kwargs)

    @wraps(raw_step_fn)
    def timed_step(*a, **kw):
        t0 = time.perf_counter()
        state_out, metrics_out = raw_step_fn(*a, **kw)
        # Force device sync: block on mean of one param leaf of state_out.
        try:
            jax.block_until_ready(jax.tree.leaves(state_out.params)[0])
        except Exception:
            pass
        t1 = time.perf_counter()
        train_timer.record('jit_step.factorized_lattice_vae', t1 - t0, True)
        return state_out, metrics_out

    return timed_step


trainer_mod._make_train_step_factorized_lattice_vae = _make_train_step_factorized_timed


# Wrap the validation evaluator to know per-epoch val cost.
trainer_mod._evaluate = _wrap(
    'trainer.validate_epoch',
    trainer_mod._evaluate,
    train_timer,
    keep_per_call=True,
)


# Macro: wrap `train_and_evaluate` itself, then `run_full_evaluation`.
# Must run BEFORE we import `run_lattice_step10_experiments`, because that
# module binds `from train.trainer import train_and_evaluate` at import time.
trainer_mod.train_and_evaluate = _wrap(
    'macro.train_and_evaluate',
    trainer_mod.train_and_evaluate,
    macro_timer,
)


# ---------- Monkey-patch evaluation side ---------- #

import eval.analysis as analysis_mod  # noqa: E402
import eval.metrics as metrics_mod  # noqa: E402

for _fn_name in (
    'compute_reconstruction_error',
    'encode_dataset',
    'check_periodicity',
    'check_modular_invariance',
    'compute_j_correlation',
    'compute_quotient_chart_quality',
    'compute_factorized_consistency',
):
    if hasattr(metrics_mod, _fn_name):
        setattr(
            metrics_mod,
            _fn_name,
            _wrap(f'metric.{_fn_name}', getattr(metrics_mod, _fn_name), eval_timer),
        )
    # analysis.py imports the names directly (`from eval.metrics import ...`),
    # so we must also replace them in analysis_mod's namespace.
    if hasattr(analysis_mod, _fn_name):
        setattr(
            analysis_mod,
            _fn_name,
            _wrap(f'metric.{_fn_name}', getattr(analysis_mod, _fn_name), eval_timer),
        )


analysis_mod._run_lattice_evaluation = _wrap(
    'eval.lattice_evaluation_block',
    analysis_mod._run_lattice_evaluation,
    eval_timer,
)

# Macro: wrap `run_full_evaluation`. Same reason as above — patch before
# `run_lattice_step10_experiments` captures the reference.
analysis_mod.run_full_evaluation = _wrap(
    'macro.run_full_evaluation',
    analysis_mod.run_full_evaluation,
    macro_timer,
)


# ---------- Optional: jax.profiler trace ---------- #

_trace_dir = os.path.join(PROFILE_OUTDIR, 'jax_trace')
if PROFILE_TRACE:
    # Start trace after first JIT compile has happened, so traces are smaller.
    # Simplest: start before any training batch and stop when train_and_evaluate returns.
    os.makedirs(_trace_dir, exist_ok=True)
    jax.profiler.start_trace(_trace_dir)
    print(f'[profile_step10] jax.profiler trace → {_trace_dir}')


# ---------- Build the short sweep ---------- #

import run_lattice_step10_experiments as step10_mod  # noqa: E402


def _short_experiments():
    first_name, first_factory = step10_mod._default_experiments()[0]

    def patched_factory():
        cfg = first_factory()
        cfg.train.num_epochs = PROFILE_EPOCHS
        cfg.train.patience = PROFILE_PATIENCE
        cfg.train.log_every = 1
        return cfg

    short_name = f'{first_name}__profile_e{PROFILE_EPOCHS}'
    return [(short_name, patched_factory)]


def main() -> int:
    print(f'[profile_step10] pid={os.getpid()} epochs={PROFILE_EPOCHS} trace={PROFILE_TRACE}')
    print(f'[profile_step10] outdir={PROFILE_OUTDIR}')

    overall_t0 = time.perf_counter()
    try:
        step10_mod.run_all(
            base_dir=os.path.dirname(PROFILE_WORKDIR) or 'runs',
            experiments=_short_experiments(),
            summary_filename='profile_step10_summaries.json',
            report_path=os.path.join(PROFILE_OUTDIR, 'short_report.md'),
        )
    finally:
        overall_t1 = time.perf_counter()
        if PROFILE_TRACE:
            try:
                jax.profiler.stop_trace()
                print(f'[profile_step10] jax.profiler trace stopped → {_trace_dir}')
            except Exception as e:
                print(f'[profile_step10] WARN: stop_trace failed: {e}')

        macro = {
            'overall_seconds': overall_t1 - overall_t0,
            'overall_start': overall_t0,
            'overall_end': overall_t1,
            'epochs': PROFILE_EPOCHS,
        }
        with open(os.path.join(PROFILE_OUTDIR, 'macro_timings.json'), 'w') as f:
            json.dump(macro, f, indent=2)
        train_timer.dump(os.path.join(PROFILE_OUTDIR, 'train_step_timings.json'))
        eval_timer.dump(os.path.join(PROFILE_OUTDIR, 'eval_step_timings.json'))
        print(f'[profile_step10] total wall = {overall_t1 - overall_t0:.2f}s')
        print(f'[profile_step10] raw timings dumped under {PROFILE_OUTDIR}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
