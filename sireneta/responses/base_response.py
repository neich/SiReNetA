from sireneta import io_helpers
import numpy as np


class BaseResponse:
    def __init__(self):
        self._directed = False
        self.con = None
        self.N = 0

    @property
    def directed(self):
        return self._directed

    def configure(self, **kwargs):
        for arg, value in kwargs.items():
            if hasattr(self, arg):
                setattr(self, arg, value)

    def _check_attrs(self, *args):
        for arg in args:
            if not hasattr(self, arg):
                raise AttributeError(arg)

    def check_config(self):
        if self.con is None:
            raise Exception("Connection matrix not defined!")
        io_helpers.validate_con(self.con)
        self._directed = np.allclose(self.con, self.con.T, rtol=1e-10, atol=1e-10)
        self.N = len(self.con)
        if self.con.dtype != np.float64:
            self.con = self.con.astype(np.float64)




