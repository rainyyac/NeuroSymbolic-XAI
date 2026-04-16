% =========================================================
% ASP RULES FOR RISK CLASSIFICATION
%
% This is the "symbolic reasoning" part of the system.
% It takes logical facts and derives a risk label.
% =========================================================


% DANGEROUS CONDITIONS
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

% Vulnerable users + traffic interaction → dangerous

risk(dangerous) :-
    vulnerable_user(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.

risk(dangerous) :-
    vulnerable_user(yes),
    traffic_density(medium, confidence(C)),
    C > 0.6.

% Emergency vehicle + traffic interaction → dangerous

risk(dangerous) :-
    emergency_vehicle(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.

% Obstruction + traffic interaction → dangerous

risk(dangerous) :-
    obstruction(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.


% MODERATE CONDITIONS
% Medium density interactions → moderate

risk(moderate) :-
    pedestrian_density(medium, confidence(C1)),
    traffic_density(medium, confidence(C2)),
    C1 > 0.6,
    C2 > 0.6.

risk(moderate) :- pedestrian_density(medium, confidence(C1)),
    crosswalk(absent),
    traffic_density(medium, confidence(C2)),
    C1 > 0.6,
    C2 > 0.6.

risk(moderate) :- obstruction(yes).

risk(moderate) :- pedestrian_density(high, confidence(C1)),
    traffic_density(low, confidence(C2)),
    C1 > 0.6,
    C2 > 0.6.

% If unknown + high traffic → conservative moderate

risk(moderate) :-
    pedestrian_density(unknown),
    traffic_density(high, confidence(_)).

% Vulnerable user/Emergency vehicle/Obstruction but no traffic → moderate

risk(moderate) :-
    vulnerable_user(yes),
    traffic_density(low, confidence(C)),
    C > 0.6.

risk(moderate) :-
    emergency_vehicle(yes),
    traffic_density(low, confidence(C)),
    C > 0.6.

risk(moderate) :-
    obstruction(yes),
    traffic_density(low, confidence(C)),
    C > 0.6.


% DEFAULT: SAFE
% If nothing risky is detected → safe

risk(safe) :-
    not risk(dangerous),
    not risk(moderate).


% CONSISTENCY CONSTRAINT
% Exactly ONE risk label must be assigned

final_risk(dangerous) :- risk(dangerous).
final_risk(moderate)  :- risk(moderate), not risk(dangerous).
final_risk(safe)      :- risk(safe), not risk(moderate), not risk(dangerous).
:- not 1 { final_risk(safe); final_risk(moderate); final_risk(dangerous) } 1.