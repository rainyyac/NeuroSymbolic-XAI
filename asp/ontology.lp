% =========================================================
% TAXONOMY (Class Hierarchy)
% =========================================================

% Actors
subclass(child, pedestrian).
subclass(adult_pedestrian, pedestrian).
subclass(cyclist, vulnerable_actor).
subclass(pedestrian, vulnerable_actor).
subclass(vulnerable_actor, actor).

% Vehicles
subclass(ambulance, emergency_vehicle).
subclass(police_car, emergency_vehicle).
subclass(fire_truck, emergency_vehicle).

subclass(car, vehicle).
subclass(bus, vehicle).
subclass(truck, vehicle).

subclass(emergency_vehicle, vehicle).
subclass(vehicle, actor).

% Obstruction
subclass(traffic_cone, obstruction).
subclass(construction_barrier, obstruction).
subclass(road_debris, obstruction).
subclass(fallen_tree_branch, obstruction).
subclass(stalled_vehicle, obstruction).

% Road Infrastructure
subclass(crosswalk, pedestrian_infrastructure).
subclass(traffic_light, road_infrastructure).
subclass(lane_marking, road_infrastructure).

% =========================================================
% INFERENCE LOGIC
% =========================================================

% 1. Transitivity (Hierarchy Traversal)
% Allows a 'child' to be recognized as both a 'pedestrian' and a 'vulnerable_actor'.
is_a(Entity, Parent) :- subclass(Entity, Parent).
is_a(Entity, Grandparent) :- subclass(Entity, Parent), is_a(Parent, Grandparent).

% 2. Map CLIP Subclass Detections to the 'is_a' System
% Assuming Python outputs facts like: pedestrian_type(child, confidence(85)).
entity_at(Entity, confidence(C)) :- pedestrian_type(Entity, confidence(C)).
entity_at(Entity, confidence(C)) :- emergency_type(Entity, confidence(C)).
entity_at(Entity, confidence(C)) :- obstruction_type(Entity, confidence(C)).

% 3. Propagate Presence Up the Hierarchy
% If a specific subclass is detected, the parent class is also marked as 'present'.
present(Class, confidence(C)) :- entity_at(Entity, confidence(C)), is_a(Entity, Class).
present(Entity, confidence(C)) :- entity_at(Entity, confidence(C)).

% 4. Bridging to Risk Rules (Connecting Classes to Reasoning Atoms)
% These atoms serve as the inputs for risk_rules_with_risk_factor.lp.

% Map any detected vulnerable actor (child, cyclist, etc.) to the vulnerable_user atom
vulnerable_user(yes, confidence(C)) :- present(vulnerable_actor, confidence(C)).

% Map any detected emergency vehicle (ambulance, police, etc.) to the emergency_vehicle atom
emergency_vehicle(yes, confidence(C)) :- present(emergency_vehicle, confidence(C)).

% Map any detected obstruction (cones, debris, etc.) to the obstruction atom
obstruction(yes, confidence(C)) :- present(obstruction, confidence(C)).

% Direct mappings for Infrastructure and Density
crosswalk(Val, confidence(C)) :- crosswalk_type(Val, confidence(C)).
traffic_density(Val, confidence(C)) :- traffic_density_type(Val, confidence(C)).

% Infer pedestrian density from specific actor detections
% If any pedestrian is detected with high confidence, we consider the area occupied.
pedestrian_density(medium, confidence(C)) :- present(pedestrian, confidence(C)).
pedestrian_density(high, confidence(C)) :- present(pedestrian, confidence(C)), C > 85.