% =========================================================
% ASP RULES FOR RISK CLASSIFICATION
%
% This is the "symbolic reasoning" part of the system.
% It takes logical facts and derives a risk label.
% =========================================================


% --- DANGEROUS CONDITIONS ---
% High pedestrian density + high traffic → dangerous

risk(dangerous) :-
    pedestrian_density(high),
    traffic_density(high).


% --- MODERATE CONDITIONS ---
% Medium density interactions → moderate

risk(moderate) :-
    pedestrian_density(medium),
    traffic_density(medium).


% --- SAFE DEFAULT ---
% If nothing risky is detected → safe

risk(safe) :-
    not risk(dangerous),
    not risk(moderate).


% --- CONSISTENCY CONSTRAINT ---
% Exactly ONE risk label must be assigned

:- not 1 { risk(safe); risk(moderate); risk(dangerous) } 1.