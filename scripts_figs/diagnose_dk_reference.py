from scripts_figs.audit_dk_convergence_gamma import build_reference_red, make_fine_graph

red = build_reference_red(40)
kp, graph = make_fine_graph(red, 100)
print(f"stabilised={graph.stabilised} sweeps={graph.completion_sweeps} roots_added={graph.roots_added} complete={graph.complete_under_policy}")
for i, pair in enumerate(graph.pairs):
    forward = [d for d in pair.forward_diagnostics if d.needs_refinement]
    backward = [d for d in pair.backward_diagnostics if d.needs_refinement]
    su = sum(pair.transport.source_unmatched)
    tu = sum(pair.transport.target_unmatched)
    if forward or backward or su or tu:
        print(f"pair {i}: k/pi={kp[i]:.6f}->{kp[i+1]:.6f} complete={pair.complete_bidirectionally} source_unmatched={pair.transport.source_unmatched} target_unmatched={pair.transport.target_unmatched}")
        for d in forward:
            print(f"  F event={d.source_index} m={d.source_multiplicity} candidates={d.candidate_indices} union={d.union_dimension} coverage={d.coverage:.6f}")
        for d in backward:
            print(f"  B event={d.source_index} m={d.source_multiplicity} candidates={d.candidate_indices} union={d.union_dimension} coverage={d.coverage:.6f}")
