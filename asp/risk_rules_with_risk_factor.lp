% =========================================================
% SILENCE WARNINGS FOR DYNAMICALLY INJECTED FACTS
% =========================================================

#defined pedestrian_density/1.
#defined pedestrian_density/2.
#defined traffic_density/2.
#defined traffic_density/1.
#defined crosswalk/2.
#defined crosswalk/1.
#defined obstruction/2.
#defined obstruction/1.
#defined emergency_vehicle/2.
#defined emergency_vehicle/1.
#defined vulnerable_user/2.
#defined vulnerable_user/1.


% =========================================================
% RISK FACTOR DEFINITIONS (DANGEROUS)
% =========================================================

% TRAFFIC CONFLICT
risk_factor(traffic_conflict) :-
    pedestrian_density(high, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 60, C2 > 60.

% VULNERABLE EXPOSURE
risk_factor(vulnerable_exposure) :-
    vulnerable_user(yes, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 60, C2 > 60.

risk_factor(vulnerable_exposure) :-
    vulnerable_user(yes, confidence(C1)),
    traffic_density(medium, confidence(C2)),
    C1 > 60, C2 > 60.

% EMERGENCY SITUATION
risk_factor(emergency_situation) :-
    emergency_vehicle(yes, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 60, C2 > 60.

% INFRASTRUCTURE FAILURE
risk_factor(infrastructure_failure) :-
    pedestrian_density(medium, confidence(C1)),
    crosswalk(absent, confidence(C2)),
    traffic_density(high, confidence(C3)),
    C1 > 60, C2 > 60, C3 > 60.

% ENVIRONMENTAL HAZARD
risk_factor(environmental_hazard) :-
    obstruction(yes, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 60, C2 > 60.


% =========================================================
% DANGEROUS = HIGH-SEVERITY RISK FACTORS
% =========================================================

risk(dangerous) :- risk_factor(traffic_conflict).
risk(dangerous) :- risk_factor(vulnerable_exposure).
risk(dangerous) :- risk_factor(emergency_situation).
risk(dangerous) :- risk_factor(infrastructure_failure).
risk(dangerous) :- risk_factor(environmental_hazard).


% =========================================================
% RISK FACTOR DEFINITIONS (MODERATE)
% =========================================================

% MODERATE TRAFFIC INTERACTION
risk_factor(moderate_interaction) :-
    pedestrian_density(medium, confidence(C1)),
    traffic_density(medium, confidence(C2)),
    C1 > 60, C2 > 60.

% OBSTRUCTION WITHOUT HEAVY TRAFFIC
risk_factor(minor_obstruction) :-
    obstruction(yes, confidence(C1)),
    traffic_density(low, confidence(C2)),
    C1 > 60, C2 > 60.

% VULNERABLE USER WITHOUT HEAVY TRAFFIC
risk_factor(moderate_vulnerable_risk) :-
    vulnerable_user(yes, confidence(C1)),
    traffic_density(low, confidence(C2)),
    C1 > 60, C2 > 60.

% UNKNOWN UNCERTAINTY
risk_factor(uncertain) :-
    pedestrian_density(unknown),
    traffic_density(high, confidence(_)).


% =========================================================
% MODERATE = MEDIUM-SEVERITY FACTORS
% =========================================================

risk(moderate) :- risk_factor(moderate_interaction).
risk(moderate) :- risk_factor(minor_obstruction).
risk(moderate) :- risk_factor(moderate_vulnerable_risk).
risk(moderate) :- risk_factor(uncertain).


% =========================================================
% SAFE - DEFAULT
% =========================================================

risk(safe) :-
    not risk(dangerous),
    not risk(moderate).

% Clearing Factors for Safety Justification
cleared_factor(no_traffic_conflict) :- not risk_factor(traffic_conflict).
cleared_factor(no_vulnerable_exposure) :- not risk_factor(vulnerable_exposure).
cleared_factor(no_emergency_situation) :- not risk_factor(emergency_situation).
cleared_factor(no_infrastructure_failure) :- not risk_factor(infrastructure_failure).
cleared_factor(no_environmental_hazard) :- not risk_factor(environmental_hazard).

% =========================================================
% FINAL DECISION (PRIORITY-BASED)
% =========================================================

% dangerous dominates everything
final_risk(dangerous) :-
    risk(dangerous).

% moderate only if NOT dangerous
final_risk(moderate) :-
    risk(moderate),
    not risk(dangerous).

% safe only if nothing else
final_risk(safe) :-
    not risk(dangerous),
    not risk(moderate).

% ensure exactly ONE final result
:- not 1 { final_risk(safe); final_risk(moderate); final_risk(dangerous) } 1.


% =========================================================
% DISPLAY ATOMS FOR EXPLANATION MODULE
% =========================================================

#show risk/1.
#show final_risk/1.
#show risk_factor/1.
#show cleared_factor/1.

% Show the 2-argument facts coming from Python
#show pedestrian_density/2.
#show traffic_density/2.
#show crosswalk/2.
#show obstruction/2.
#show emergency_vehicle/2.
#show vulnerable_user/2.

% Show the 1-argument 'unknown' facts (e.g., pedestrian_density(unknown))
#show pedestrian_density/1.
#show traffic_density/1.
#show crosswalk/1.
#show obstruction/1.
#show emergency_vehicle/1.
#show vulnerable_user/1.