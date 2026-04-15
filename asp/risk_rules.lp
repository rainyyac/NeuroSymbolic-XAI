% =========================================================
% ASP RULES FOR RISK CLASSIFICATION
%
% This is the "symbolic reasoning" part of the system.
% It takes logical facts and derives a risk label.
% =========================================================


% --- DANGEROUS CONDITIONS ---
% High pedestrian density + high traffic → dangerous

risk(dangerous) :-
    pedestrian_density(high, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 0.6,
    C2 > 0.6.

risk(dangerous) :- pedestrian_density(medium, confidence(C1)),
    crosswalk(absent),
    traffic_density(high, confidence(C2)),
    C1 > 0.6,
    C2 > 0.6.

risk(dangerous) :- emergency_vehicle(yes).

risk(dangerous) :- vulnerable_user(yes).


% --- MODERATE CONDITIONS ---
% Medium density interactions → moderate

risk(moderate) :-
    pedestrian_density(medium, confidence(C1)),
    traffic_density(medium, confidence(C2))
    C1 > 0.6,
    C2 > 0.6.

risk(moderate) :- pedestrian_density(medium, confidence(C1)),
    crosswalk(absent),
    traffic_density(medium, confidence(C2))
    C1 > 0.6,
    C2 > 0.6.

risk(moderate) :- obstruction(yes).

risk(moderate) :- pedestrian_density(high, confidence(C1)),
    traffic_density(low, confidence(C2))
    C1 > 0.6,
    C2 > 0.6.

% If unknown + high traffic → conservative moderate

risk(moderate) :-
    pedestrian_density(unknown),
    traffic_density(high, confidence(_)).


% --- SAFE DEFAULT ---
% If nothing risky is detected → safe

risk(safe) :-
    not risk(dangerous),
    not risk(moderate).


% --- CONSISTENCY CONSTRAINT ---
% Exactly ONE risk label must be assigned

:- not 1 { risk(safe); risk(moderate); risk(dangerous) } 1.