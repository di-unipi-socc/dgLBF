:-['../glbf.pl'].



:- set_prolog_flag(stack_limit, 128 000 000 000).
:- set_prolog_flag(last_call_optimisation, true).

sim_glbf(Out, Alloc, Infs, Time) :-
    statistics(inferences, I1),
        statistics(cputime, T1),
            glbf(Out, Alloc), 
        statistics(cputime, T2),
    statistics(inferences, I2), 

	Infs is I2 - I1 - 5,
	Time is T2 - T1.

loadInfrastructure(Path) :-
    open(Path, read, Str),
    (retractall(node(_,_)), retractall(link(_,_,_,_)); true),
    readAndAssert(Str).

loadFlows(Path) :-
    open(Path, read, Str),
    (retractall(flow(_, _, _, _, _, _, _, _)); true),
    readAndAssert(Str).

readAndAssert(Str) :-
    read(Str, X), (X == end_of_file -> close(Str) ; assert(X), readAndAssert(Str)).