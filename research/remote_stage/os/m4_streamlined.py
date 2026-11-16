"""Public name for the bitwise-validated E095A M4 inference implementation.

The frozen implementation stays in m4_clean_deploy.py so historical source
hashes and experiment imports remain valid. This class inherits its complete
construction and forward path without changing parameter keys or numerics.
"""

if __package__:
    from .m4_clean_deploy import CleanM4
else:
    from m4_clean_deploy import CleanM4


class M4Streamlined(CleanM4):
    """M4-streamlined: three recurrent matchers and five total LSR fits.

    Construct from a strictly loaded protected M4 in eval mode. Forward
    returns raw flow after four internal fits; the caller applies the fifth,
    terminal official LSR. This is an inference-only model.
    """

    method_name = "M4-streamlined"
