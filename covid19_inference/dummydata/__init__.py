# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-05-26 13:27:31
# @Last Modified: 2020-06-17 10:01:45
# ------------------------------------------------------------------------------ #
from .model import DummyModel
from .compartmental_models import SIR, SEIR
from .spreading_rate import generate_lambda_t_from_model