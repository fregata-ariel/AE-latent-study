"""Run the Step 9 topology Phase A/B follow-up for the selected winner."""

from __future__ import annotations

import json
import os

from run_lattice_step9_experiments import _select_best_run
from run_latent_topology_diagnostics import run_all as run_topology_all
from run_topology_phaseB_comparison import run_all as run_topology_phaseb_all


DEFAULT_TOPOLOGY_OVERRIDES = {
    'eval': {
        'ph_max_samples': 1500,
        'ph_random_projection_trials': 4,
    },
}


def _load_step9_summaries(path: str) -> dict[str, dict]:
    with open(path) as f:
        return json.load(f)


def _select_step9_winner(summary_path: str) -> tuple[str, dict, bool]:
    summaries = _load_step9_summaries(summary_path)
    winner_name, winner_summary, used_gate = _select_best_run(summaries)
    if winner_name is None or winner_summary is None:
        raise RuntimeError('No Step 9 summaries found; run Step 9 experiments first.')
    return winner_name, winner_summary, used_gate


def _make_experiments(winner_name: str, config_overrides: dict) -> list[dict]:
    return [
        {
            'name': 't2_standard',
            'kind': 'control',
            'config_source': 'configs.t2_standard',
            'config_overrides': config_overrides,
        },
        {
            'name': 't2_torus',
            'kind': 'control',
            'config_source': 'configs.t2_torus',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_standard_norm_inv',
            'kind': 'lattice',
            'config_source': 'configs.lattice_standard_norm_inv',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_vae_norm_beta001',
            'kind': 'lattice',
            'config_source': 'configs.lattice_vae_norm_beta001',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_vae_norm_inv_b010_l100',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_vae_norm_inv_b030_l100',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_vae_wide_norm_inv_b003_l030',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_factorized_vae_fd_b030_q100_g030_d030',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_factorized_vae_fd_b030_q100_g030_d030_l020_s030',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': 'lattice_factorized_vae_fd_b030_q100_g030_d030_j010_ld010_r100',
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
        {
            'name': winner_name,
            'kind': 'lattice',
            'config_overrides': config_overrides,
        },
    ]


def run_all(
    base_dir: str = 'runs',
    step9_summary_path: str = 'runs/lattice_step9_summaries.json',
    diagnostics_dir: str = 'runs/topology_diagnostics_step9',
    phasea_report_path: str = 'walkthrough-topology-step9-phaseA.md',
    phaseb_report_path: str = 'walkthrough-topology-step9-phaseB.md',
    roadmap_path: str = 'ae-latent-study-roadmap-step9.md',
    config_overrides: dict | None = None,
) -> dict:
    """Run topology Phase A/B follow-up for the selected Step 9 winner."""
    os.makedirs(diagnostics_dir, exist_ok=True)
    config_overrides = config_overrides or DEFAULT_TOPOLOGY_OVERRIDES
    winner_name, winner_summary, used_gate = _select_step9_winner(step9_summary_path)
    experiments = _make_experiments(winner_name, config_overrides)

    phasea = run_topology_all(
        base_dir=base_dir,
        diagnostics_dir=diagnostics_dir,
        report_path=phasea_report_path,
        experiments=experiments,
    )
    phaseb = run_topology_phaseb_all(
        base_dir=base_dir,
        diagnostics_dir=diagnostics_dir,
        summary_filename='phaseB_comparison_summary.json',
        report_path=phaseb_report_path,
        roadmap_path=roadmap_path,
        experiments=experiments,
        focus_run_name=winner_name,
    )
    combined = {
        'winner_name': winner_name,
        'winner_used_gate': used_gate,
        'winner_summary': winner_summary,
        'config_overrides': config_overrides,
        'phaseA': phasea,
        'phaseB': phaseb,
    }
    with open(os.path.join(diagnostics_dir, 'step9_followup_summary.json'), 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nSelected Step 9 winner: {winner_name}")
    print(f"Gate status: {'passed' if used_gate else 'fallback winner (no gate pass)'}")
    print(f"Phase A report: {phasea_report_path}")
    print(f"Phase B report: {phaseb_report_path}")
    print(f"Roadmap: {roadmap_path}")
    return combined


if __name__ == '__main__':
    run_all()
