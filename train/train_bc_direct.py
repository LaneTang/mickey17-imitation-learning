"""Train BC to predict the leader's current six-joint command a[t].

Preview is the default. Add --train to create a separate bc_direct checkpoint.
"""

from bc_common import main


if __name__ == "__main__":
    main("direct")
