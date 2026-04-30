% =========================================================
% TAXONOMY (Class Hierarchy)
% =========================================================

% Actors
subclass(pedestrian, vulnerable_actor).
subclass(cyclist, vulnerable_actor).
subclass(child, pedestrian).
subclass(vulnerable_actor, actor).

% Vehicles
subclass(ambulance, emergency_vehicle).
subclass(police_car, emergency_vehicle).
subclass(emergency_vehicle, vehicle).
subclass(car, vehicle).
subclass(bus, vehicle).
subclass(vehicle, actor).


% =========================================================
% INFERENCE LOGIC
% =========================================================

% Transitivity
is_a(Entity, Parent) :- subclass(Entity, Parent).
is_a(Entity, Grandparent) :- subclass(Entity, Parent), is_a(Parent, Grandparent).

% Connecting CLIP facts to Ontological classes.
entity_present(pedestrian) :- pedestrian_density(high, _).
entity_present(pedestrian) :- pedestrian_density(medium, _).
entity_present(emergency_vehicle) :- emergency_vehicle(yes, _).
entity_present(vulnerable_actor) :- vulnerable_user(yes, _).
entity_present(obstruction) :- obstruction(yes, _).