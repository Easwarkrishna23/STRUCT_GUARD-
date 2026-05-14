"""Central configuration for all experiments."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    dropout_rate: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 20


@dataclass
class AttackConfig:
    # Experiment acceptance gate. Relative target means attacked accuracy must be
    # <= 50% of baseline accuracy, i.e. baseline - attacked >= 0.5 * baseline.
    # This remains stricter than the current command's >=30% degradation floor.
    target_accuracy_drop: float = 0.0
    target_drop_fraction: float = 0.50
    enforce_target_drop: bool = True
    thesis_acceptance_mode: bool = True

    # Nettack — increased perturbations per node for stronger targeted impact
    nettack_n_perturbations: int = 20
    nettack_max_perturbations: int = 40
    nettack_direct: bool = True

    # Meta Attack — 20% budget drives accuracy into 40-60% range
    meta_epochs: int = 200
    meta_lr: float = 0.1
    meta_budget_ratio: float = 0.20
    structural_max_budget_ratio: float = 0.30

    # Random Structure — 25% budget for strong baseline attack
    random_budget_ratio: float = 0.25

    # Feature Perturbation — ε=0.5 drives ~40% accuracy drop on Cora BoW features
    feature_epsilon: float = 0.5

    # Edge Flip — 20% budget to match structural attack strength
    edge_flip_budget_ratio: float = 0.20

    # Gradient-Based — ε=0.15 lands in 40-60% drop range (ε=0.3 was too extreme at 0%)
    grad_epsilon: float = 0.15
    grad_steps: int = 20

    # Surrogate loss / high-confidence targeting.
    high_confidence_quantile: float = 0.70
    high_confidence_weight: float = 2.0

    # Feature caps for realistic calibration.
    cora_feature_flip_cap: float = 0.50
    elliptic_quantile_clip_low: float = 0.01
    elliptic_quantile_clip_high: float = 0.99

    def required_drop(self, baseline_acc: float) -> float:
        return max(self.target_accuracy_drop, baseline_acc * self.target_drop_fraction)


@dataclass
class DefenseConfig:
    # Edge Pruning — percentile-based: remove bottom prune_pct% of edges by cosine sim
    # This is dataset-agnostic (works regardless of absolute sim values)
    prune_percentile: float = 10.0    # remove bottom 10% least-similar edges
    cosine_threshold: float = 0.0     # fallback fixed threshold (used if prune_percentile=0)
    min_edges_ratio: float = 0.7      # keep at least 70% of original edges

    # Graph Reconstruction (k-NN)
    knn_k: int = 3

    # STRUC-GUARD+ and ontology thresholds.
    centrality_quantile: float = 0.90
    centrality_cosine_threshold: float = 0.20
    topic_jaccard_threshold: float = 0.15
    degree_z_threshold: float = 2.0
    degree_target_z: float = 1.0
    small_world_apl_tolerance: float = 0.15
    topic_top_k: int = 20

    # Scale-Free Integrity Engine thresholds.
    preferential_low_degree_quantile: float = 0.35
    preferential_attachment_threshold: float = 0.10
    assortativity_cosine_threshold: float = 0.15
    assortativity_degree_gap_z: float = 1.50
    eigenvector_top_quantile: float = 0.90
    bose_einstein_fitness_threshold: float = 0.25
    gamma_tolerance: float = 0.35
    hierarchical_slope_target: float = -1.0
    hierarchical_slope_tolerance: float = 0.65
    path_length_reduction_epsilon: float = 0.10
    temporal_drift_z_threshold: float = 3.0
    suspicious_node_edge_keep_ratio: float = 0.10

    # Adversarial retraining augmentation.
    adv_feature_epsilon: float = 0.05
    adv_edge_drop_rate: float = 0.03
    adv_edge_add_rate: float = 0.03

    # Validation-selected recovery candidates. This stays realistic: the final
    # test score is reported only after choosing by validation accuracy.
    adaptive_recovery_candidates: bool = True
    smoothing_steps: int = 2
    smoothing_residual_alpha: float = 0.25
    trusted_baseline_fallback: bool = True


@dataclass
class DynamicGraphConfig:
    """SBM-based temporal graph parameters."""
    num_nodes: int = 2708          # match Cora size
    num_communities: int = 7       # match Cora classes
    timesteps: int = 10
    p_in: float = 0.008            # intra-community edge prob (tuned to match Cora density ~5K edges)
    p_out: float = 0.0002          # inter-community edge prob
    feature_dim: int = 1433        # match Cora feature dim
    community_switch_rate: float = 0.02   # fraction of nodes that switch community per step
    edge_change_rate: float = 0.05        # fraction of edges added/removed per step
    feature_noise_std: float = 0.01       # Gaussian noise added to features per step
    seed: int = 42


@dataclass
class Config:
    experiment_version: str = "scale-free-integrity-engine-v1"
    seed: int = 42
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    figures_dir: Path = field(default_factory=lambda: Path("results/figures"))
    tables_dir: Path = field(default_factory=lambda: Path("results/tables"))
    checkpoints_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    dynamic: DynamicGraphConfig = field(default_factory=DynamicGraphConfig)

    def make_dirs(self):
        for d in [self.data_dir, self.results_dir, self.figures_dir,
                  self.tables_dir, self.checkpoints_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def signature(self) -> str:
        """Stable hash for cache invalidation when experiment knobs change."""
        payload = {
            "experiment_version": self.experiment_version,
            "model": asdict(self.model),
            "attack": asdict(self.attack),
            "defense": asdict(self.defense),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


# Singleton used throughout the project
cfg = Config()
