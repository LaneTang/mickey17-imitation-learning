"""Train BC to predict current command minus observed follower pose: a[t] - s[t].

Preview is the default. Add --train to create a separate bc_residual checkpoint.
"""

from bc_common import main


if __name__ == "__main__":
    main("residual")
