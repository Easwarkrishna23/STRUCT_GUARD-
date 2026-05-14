# Algorithm 2: Scale-Free Integrity Engine

**Input:** Poisoned graph \(G'=(V,E',X,T)\), baseline GNN \(f_\theta\), validation split, previous Elliptic snapshot \(G_{t-1}\) when available, thresholds \(\tau_c,\tau_j,\epsilon,\gamma\).  
**Output:** Defended graph \(\hat G=(V,\hat E,\hat X,T)\), suspicious edge/node sets, and recovery diagnostics.

1. Estimate clean graph priors: degree sequence, edge homophily, assortativity, clustering spectrum \(C(k)\), sampled average path length \(L\), and power-law degree exponent \(\gamma\).
2. Build node topic sets:
   - Cora: nonzero binary bag-of-words indices.
   - Elliptic: top absolute z-score transaction features.
3. Flag `SuspiciousEdge` when topic Jaccard is below \(\tau_j\), cosine similarity is low, preferential attachment likelihood is weak, or the edge creates hub-outlier disassortativity.
4. For Elliptic, compare the current snapshot with \(G_{t-1}\). Flag `SuspiciousNode` when robust feature drift exceeds the temporal z-score threshold.
5. Run Scale-Free pruning:
   - remove low-cosine high-betweenness bridge edges;
   - prune semantically unrelated low-degree to low-degree edges with low growth probability;
   - prune hub-outlier edges with cosine \(<0.15\);
   - for eigenvector-central nodes, prune low Bose-Einstein-fitness incident edges only when the resulting degree exponent remains within tolerance;
   - isolate temporal-drift nodes by preserving only their highest-fitness incident edges.
6. Denoise node attributes by residual neighborhood smoothing on the pruned topology.
7. Reconstruct kNN edges only if they improve endpoint clustering, maintain \(C(k)\sim k^{-1}\), and do not reduce sampled average path length by more than \(\epsilon\).
8. Adversarially retrain the GNN on the defended graph with training-node feature PGD and edge perturbation augmentations.
9. Select the defended candidate using validation accuracy only; report final test recovery metrics once.
10. Accept the run only if attack degradation passes the configured gate and defended test accuracy is at least the clean baseline.
