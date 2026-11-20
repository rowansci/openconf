"""Regression tests for low-mode source selection after initial pool eviction."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from openconf import ConformerConfig, generate_conformers
from openconf.propose import hybrid


def test_low_mode_sources_survive_initial_seed_insertion() -> None:
    """Generate low-mode children after seed insertion evicts original seeds."""
    ensemble = generate_conformers(
        "CCCCCCCC",
        config=ConformerConfig(
            n_seeds=8,
            seed_prune_rms_thresh=-1.0,
            n_steps=0,
            max_out=1,
            pool_max=1,
            random_seed=42,
            num_threads=1,
            use_low_mode_following=True,
            low_mode_n_source_seeds=8,
            do_final_refine=False,
        ),
    )
    assert ensemble.n_conformers == 1


def test_low_mode_sources_are_pool_members(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select surviving seeds before applying source count limit."""
    visited: list[int] = []
    seed_ids: list[int] = []

    def generate_seeds(
        self: hybrid.HybridProposer, n_seeds: int, prune_rms_thresh: float = -1.0
    ) -> list[tuple[int, float]]:
        seed_ids.extend(AllChem.EmbedMultipleConfs(self.mol, numConfs=n_seeds, randomSeed=42))
        return [(cid, float(len(seed_ids) - i)) for i, cid in enumerate(seed_ids)]

    def low_mode_seeds(
        mol: Chem.Mol, ff_props: object, conf_id: int, minimizer: object, **kwargs: object
    ) -> list[tuple[int, float]]:
        mol.GetConformer(conf_id)
        visited.append(conf_id)
        return []

    monkeypatch.setattr(hybrid.HybridProposer, "generate_seeds", generate_seeds)
    monkeypatch.setattr(hybrid, "generate_low_mode_seeds", low_mode_seeds)
    ensemble = generate_conformers(
        "CCCCCCCC",
        config=ConformerConfig(
            n_seeds=3,
            n_steps=0,
            max_out=1,
            pool_max=1,
            random_seed=42,
            num_threads=1,
            use_low_mode_following=True,
            low_mode_n_source_seeds=3,
            do_final_refine=False,
        ),
    )
    assert len(seed_ids) == 3
    assert visited == [seed_ids[-1]]
    assert ensemble.n_conformers == 1
