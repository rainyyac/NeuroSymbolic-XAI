% =========================================================
% RISK FACTOR DEFINITIONS (DANGEROUS)
% =========================================================

% TRAFFIC CONFLICT
risk_factor(traffic_conflict) :-
    pedestrian_density(high, confidence(C1)),
    traffic_density(high, confidence(C2)),
    C1 > 0.6, C2 > 0.6.

% VULNERABLE EXPOSURE
risk_factor(vulnerable_exposure) :-
    vulnerable_user(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.

risk_factor(vulnerable_exposure) :-
    vulnerable_user(yes),
    traffic_density(medium, confidence(C)),
    C > 0.6.

% EMERGENCY SITUATION
risk_factor(emergency_situation) :-
    emergency_vehicle(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.

% INFRASTRUCTURE FAILURE
risk_factor(infrastructure_failure) :-
    pedestrian_density(medium, confidence(C1)),
    crosswalk(absent),
    traffic_density(high, confidence(C2)),
    C1 > 0.6, C2 > 0.6.

% ENVIRONMENTAL HAZARD
risk_factor(environmental_hazard) :-
    obstruction(yes),
    traffic_density(high, confidence(C)),
    C > 0.6.


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
    C1 > 0.6,
    C2 > 0.6.

% OBSTRUCTION WITHOUT HEAVY TRAFFIC
risk_factor(minor_obstruction) :-
    obstruction(yes),
    traffic_density(low, confidence(C)),
    C > 0.6.

% VULNERABLE USER WITHOUT HEAVY TRAFFIC
risk_factor(moderate_vulnerable_risk) :-
    vulnerable_user(yes),
    traffic_density(low, confidence(C)),
    C > 0.6.

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

#show vulnerable_user/1.
#show vulnerable_user_confidence/2.

#show crosswalk/1.
#show crosswalk_confidence/2.

#show obstruction/1.
#show obstruction_confidence/2.

#show emergency_vehicle/1.
#show emergency_vehicle_confidence/2.

#show risk/1.
#show final_risk/1.