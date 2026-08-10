# This is an example script to get all the affordance information specified in xml files.

import LIBERO.scripts.init_path as init_path
from LIBERO.libero.libero.envs.objects import OBJECTS_DICT
from LIBERO.libero.libero.utils.object_utils import get_affordance_regions

affordances = get_affordance_regions(OBJECTS_DICT)

print(affordances)
