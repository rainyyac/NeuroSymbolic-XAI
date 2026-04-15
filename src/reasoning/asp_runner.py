"""
ASP RUNNER

This module connects Python with the ASP solver (clingo).

It:
- loads rules
- adds facts
- computes the final result
"""

from clingo import Control


def run_asp(facts, rules_path="asp/risk_rules.lp"):
    """
    Runs ASP reasoning.

    Args:
        facts (list): list of ASP facts (strings)
        rules_path (str): path to rule file

    Returns:
        risk_label (str): safe / moderate / dangerous
    """

    ctl = Control()

    # Load ASP rules from file
    with open(rules_path, "r") as f:
        ctl.add("base", [], f.read())

    # Add facts dynamically
    for fact in facts:
        ctl.add("base", [], fact)

    # Ground the program (prepare for solving)
    ctl.ground([("base", [])])

    result = None

    # Solve the logic program
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            # Extract atoms from solution
            for atom in model.symbols(shown=True):
                if atom.name == "risk":
                    result = atom.arguments[0].name

    return result