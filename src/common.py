from dataclasses import dataclass, field
import collections

StepResult = collections.namedtuple(
    "StepResult", field_names=["obs", "reward", "done", "info"]
)


@dataclass
class History:
    g: list = field(default_factory=list)
    t: list = field(default_factory=list)
    p: list = field(default_factory=list)
    v: list = field(default_factory=list)
    i: list = field(default_factory=list)
    dp: list = field(default_factory=list)
    dv: list = field(default_factory=list)
    di: list = field(default_factory=list)
    g_norm: list = field(default_factory=list)
    t_norm: list = field(default_factory=list)
    p_norm: list = field(default_factory=list)
    v_norm: list = field(default_factory=list)
    i_norm: list = field(default_factory=list)
    deg: list = field(default_factory=list)