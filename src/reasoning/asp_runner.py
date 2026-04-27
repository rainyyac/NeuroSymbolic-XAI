"""
ASP RUNNER

This module connects Python with the ASP solver (clingo).

It:
- loads rules
- adds facts
- computes the final result
"""

from clingo import Control
import pathlib
DEFAULT_RULES = pathlib.Path(__file__).parent.parent.parent / "asp" / "risk_rules_with_risk_factor.lp"

def run_asp(facts, rules_path=DEFAULT_RULES):
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
    #with open(rules_path, "r") as f:
    #    ctl.add("base", [], f.read())

    # Load ASP rules from file
    try:
        # 1. Force UTF-8 encoding to prevent invisible character crashes
        with open(rules_path, "r", encoding="utf-8") as f:
            rules_text = f.read()

        # 2. Add to clingo
        ctl.add("base", [], rules_text)

    except UnicodeDecodeError as e:
        print("\n[FILE ERROR]: Your .lp file contains invalid characters (likely copy-paste artifacts).")
        raise e
    except RuntimeError as e:
        print("\n[CLINGO ERROR]: There is a syntax error in your .lp file!")
        raise e
    except Exception as e:
        print(f"\n[UNKNOWN ERROR on add/read]: {e}")
        raise e

    # Add facts dynamically
    for fact in facts:
        ctl.add("base", [], fact)

    # Ground the program (prepare for solving)
    ctl.ground([("base", [])])

    answer_set = []
    model_found = False

    # Solve the logic program
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            model_found = True
            # Return strings for all atoms marked with #show
            answer_set = [str(atom) for atom in model.symbols(shown=True)]

    if not model_found:
        raise RuntimeError("ASP Error: The logic program is UNSATISFIABLE. "
                           "Check for conflicting rules or violated integrity constraints.")

    return answer_set