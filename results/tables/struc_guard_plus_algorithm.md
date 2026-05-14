# Algorithm 1: STRUC-GUARD+ with Ontology-Driven Self-Healing

**Input:** GNN model `f = (MSG, AGG, UPD)`; attacked graph `G' = (V, E', X)`; normalized adjacency `A'`; labels available on train/validation masks; trainable parameters `Theta`; defense thresholds `tau_c`, `tau_J`, `tau_d`; memory coefficient `beta`.

**Output:** defended graph `G* = (V, E*, X*)`; robust parameters `Theta*`.

```text
Initialize h_u^0 = x_u for every node u in V
Initialize edge memory omega_uv^0 = 1 for every edge (u, v) in E'

Build topic set T_u for every node u
    if X is binary: T_u = {i | x_ui > 0}
    otherwise:     T_u = top-k absolute z-score feature indices

for each edge (u, v) in E' do
    s_uv = cosine(x_u, x_v)
    J_uv = |T_u intersect T_v| / |T_u union T_v|
    B_uv = edge_betweenness(u, v)
    if J_uv < tau_J or (topic_mismatch(u, v) and s_uv < tau_c) then
        mark (u, v) as SuspiciousEdge
    end
end

for each edge (u, v) in E' do
    if (u, v) is SuspiciousEdge or (B_uv is high and s_uv < tau_c) then
        remove (u, v) from E'
    end
end

for each node u in V with anomalous degree do
    for each incident edge (u, v) do
        phi_uv = 0.50*cosine(x_u, x_v)
                 + 0.30*Jaccard(T_u, T_v)
                 + 0.20*degree_consistency(u, v)
    end
    prune lowest-fitness incident edges until degree(u) returns near power-law mean
end

X_s = residual_smooth(A_pruned, X)

for each node u in V do
    C_u = top-k nearest neighbors of u under cosine(X_s)
    for each v in C_u do
        add (u, v) only if local_clustering improves
        and sampled average path length remains within small-world tolerance
    end
end

for layer k = 1 to K do
    for node u in V do
        for neighbor v in N_u* do
            alpha_uv^k = cosine(h_u^k, h_v^k)
            c_uv^k = [alpha_uv^k, centrality_uv, Jaccard(T_u, T_v), phi_uv]
            alpha_hat_uv^k = sigmoid(c_uv^k W)
            omega_uv^k = beta*omega_uv^(k-1) + (1-beta)*alpha_hat_uv^k
            m_uv^k = MSG'(h_u^k, h_v^k, A_uv)
        end
        m_hat_u^k = AGG'({omega_uv^k * m_uv^k : v in N_u*})
        h_u^(k+1) = UPD'(h_u^k, m_hat_u^k)
    end
end

Train Theta* on G* using adversarial feature perturbation
and edge drop/add augmentation selected by validation accuracy

return G*, Theta*
```

**Acceptance gate:** an attack is accepted only when `baseline_acc - attacked_acc >= 0.50 * baseline_acc`; a defense is accepted only when `defended_acc >= baseline_acc`.
