% node(NodeId, MinimumLatencyBudget)
% TODO: estimate MinimumLatencyBudget with queue model (i.e. M/M/1) or UBS calculus?
node(nStart, 8).
node(n1, 5).
node(n2, 10).
node(nEnd, 5).

% link(N1,N2, TProp [ms], Bandwidth [Mbps])
link(nStart, n1, 40, 100).
link(nStart, n2, 15, 20).
link(nStart, n3, 410, 100).
link(nStart, n4, 3, 100).
link(nStart, n5, 20, 100).
link(n1, nEnd, 3, 20).
link(n2, nEnd, 6, 50).

% flow(FlowId, NStart, NEnd, PacketSize [Mb], BurstSize [#packets], BitRate [Mbps], E2ELatencyBudget [ms], TolerationThreshold [ms])
flow(f1, nStart, nEnd, 0.008, 3, 5, 50, 10).
flow(f2, nStart, nEnd, 0.01, 4, 10, 100, 10).
flow(f3, nStart, nEnd, 0.008, 5, 8, 40, 10).

candidate(_, [('nStart', 'n1'), ('n1', 'nEnd')]).

degree(nStart, 2).
degree(n1, 1).
degree(n2, 1).
degree(nEnd, 0).
