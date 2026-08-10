from typing import List

import numpy as np


class Expression:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class UnaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1):
        raise NotImplementedError


class BinaryAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, arg1, arg2):
        raise NotImplementedError


class MultiarayAtomic(Expression):
    def __init__(self):
        pass

    def __call__(self, *args):
        raise NotImplementedError


class TruePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return True


class FalsePredicateFn(MultiarayAtomic):
    def __init__(self):
        super().__init__()

    def __call__(self, *args):
        return False


class InContactPredicateFn(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg1.check_contact(arg2)


class In(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_contact(arg1) and arg2.check_contain(arg1)


class On(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return arg2.check_ontop(arg1)

        # if arg2.object_state_type == "site":
        #     return arg2.check_ontop(arg1)
        # else:
        #     obj_1_pos = arg1.get_geom_state()["pos"]
        #     obj_2_pos = arg2.get_geom_state()["pos"]
        #     # arg1.on_top_of(arg2) ?
        #     # TODO (Yfeng): Add checking of center of mass are in the same regions
        #     if obj_1_pos[2] >= obj_2_pos[2] and arg2.check_contact(arg1):
        #         return True
        #     else:
        #         return False


class Up(BinaryAtomic):
    def __call__(self, arg1):
        return arg1.get_geom_state()["pos"][2] >= 1.0


class Stack(BinaryAtomic):
    def __call__(self, arg1, arg2):
        return (
            arg1.check_contact(arg2)
            and arg2.check_contain(arg1)
            and arg1.get_geom_state()["pos"][2] > arg2.get_geom_state()["pos"][2]
        )


class LeftOfX(BinaryAtomic):
    """True iff arg1 is to the left of arg2 along the table's X axis, beyond a margin."""

    MARGIN = 0.03

    def __call__(self, arg1, arg2):
        arg1_pos = arg1.get_geom_state()["pos"]
        arg2_pos = arg2.get_geom_state()["pos"]
        return (arg2_pos[0] - arg1_pos[0]) > self.MARGIN


class RightOfX(BinaryAtomic):
    """True iff arg1 is to the right of arg2 along the table's X axis, beyond a margin."""

    MARGIN = 0.03

    def __call__(self, arg1, arg2):
        arg1_pos = arg1.get_geom_state()["pos"]
        arg2_pos = arg2.get_geom_state()["pos"]
        return (arg1_pos[0] - arg2_pos[0]) > self.MARGIN


class NearTo(BinaryAtomic):
    """True iff the planar (XY) distance between arg1 and arg2 is below a threshold."""

    THRESHOLD = 0.22

    def __call__(self, arg1, arg2):
        arg1_pos = arg1.get_geom_state()["pos"]
        arg2_pos = arg2.get_geom_state()["pos"]
        return np.linalg.norm(arg1_pos[:2] - arg2_pos[:2]) < self.THRESHOLD


class FarFrom(BinaryAtomic):
    """True iff the planar (XY) distance between arg1 and arg2 is above a threshold."""

    THRESHOLD = 0.20

    def __call__(self, arg1, arg2):
        arg1_pos = arg1.get_geom_state()["pos"]
        arg2_pos = arg2.get_geom_state()["pos"]
        return np.linalg.norm(arg1_pos[:2] - arg2_pos[:2]) > self.THRESHOLD


class PrintJointState(UnaryAtomic):
    """This is a debug predicate to allow you print the joint values of the object you care"""

    def __call__(self, arg):
        print(arg.get_joint_state())
        return True


class Open(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_open()


class Close(UnaryAtomic):
    def __call__(self, arg):
        return arg.is_close()


class TurnOn(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_on()


class TurnOff(UnaryAtomic):
    def __call__(self, arg):
        return arg.turn_off()
