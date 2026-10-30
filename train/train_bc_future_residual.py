"""Train BC to predict the 3-frame future command minus current pose: a[t+3] - s[t].

Future labels never cross episode boundaries. Add --train for an independent checkpoint.
"""

from bc_common import main


if __name__ == "__main__":
    main("future-residual")
