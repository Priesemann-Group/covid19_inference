# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-05-26 13:27:31
# @Last Modified: 2020-06-04 14:54:45
# ------------------------------------------------------------------------------ #
from .model import DummyModel
from .compartmental_models import SIR, SEIR
from .spreading_rate import lambda_t_with_sigmoids