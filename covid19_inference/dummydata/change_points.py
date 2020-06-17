# ------------------------------------------------------------------------------ #
# @Author:        Sebastian B. Mohr
# @Email:         
# @Created:       2020-06-04 15:48:54
# @Last Modified: 2020-06-17 11:02:26
# ------------------------------------------------------------------------------ #


"""
This file holds a class for changepoints
"""
from types import *
import datetime
import numpy as np



class ChangePoint(object):
    """docstring for ChangePoint"""


    modulation_types = ["step","sigmoid"]

    def __init__(self, lambda_before, lambda_after, date_begin, length, modulation="sigmoid"):
        assert isinstance(date_begin, datetime.datetime), f"date_begin_cp has to be datetime"
        assert isinstance(length, datetime.timedelta), f"len_cp has to be timedelta"
        assert modulation in self.modulation_types, f"Possible modulation types are {self.modulation_types}"

        self._lambda_after = lambda_after
        self._lambda_before = lambda_before
        self._date_begin = date_begin
        self._length = length
        self.modulation = modulation

    # ------------------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------------------ #

    @property
    def lambda_after(self):
        """
            The lambda value after the change point happened!
        """
        return self._lambda_after
    @property
    def lambda_before(self):
        """
            The lambda value before the change point happened!
        """
        return self._lambda_before

    @property
    def date_begin(self):
        """
        The date on which the change point begins.
        """
        return self._date_begin
    
    @property
    def date_end(self):
        """
            The date on which the change point ends.
        """
        return self._date_begin + self._length
    
    @property
    def date_len(self):
        """
            The length of the changepoint
        """
        return self._length

    @property
    def num_len(self):
        """
            Returns length of changepoint in microseconds
        """
        return self._length/datetime.timedelta(days=1)


    # ------------------------------------------------------------------------------ #
    # Modulation
    # ------------------------------------------------------------------------------ #
    def get_lambdas(self,dt):
        """
            Returns the lambda t array starting from change point begin to change point end
            i.e. as long as the change point length.

            Parameters
            ----------
            dt: number
                Step size of the returned lambda_t array corresponding to the model resolution.
                In days!
        """

        # 1. Create t array for the corresponding time range, choose microseconds as resolution
        t = np.arange(0,self.num_len,dt)
        if self.modulation == "step":
            lambda_t = self._step_func(t)
        elif self.modulation == "sigmoid":
            lambda_t = self._sigmoid_func(t)
        return lambda_t

    def _step_func(self,t):
        """
            Computes the step function for our change point
        """

        #If change point length is 0 return new value
        if self.num_len == 0:
            return [self.lambda_after]*len(t.flatten())

        slope = (self.lambda_before-self.lambda_after)/(-self.num_len)

        return slope*t+self.lambda_before


    def _sigmoid_func(self,t):
        """
            Computes sigmoidal lambda values between date begin and date end.
        """
        #If change point length is 0 return new value

        def logistics_from_cps(x, x_1, y_1, x_2, y_2):
            """
                Calculates the lambda value at the point x
                between cp1_x and cp2_x.
                
                k is scaled with the inverse length of the change point
            """
            #Convert k value to seconds
            k_d = 1/self.date_len.days * 15 #in days 

            L = y_2-y_1
            C = y_1
            x_0 = np.abs(x_1 - x_2)/2 + x_1
            return L/(1+np.exp(-k_d*(x-x_0)))+C


        if self.num_len == 0:
            return [self.lambda_after]*len(t.flatten())        

        x_1 = 0 
        y_1 = self.lambda_before

        x_2 = self.num_len
        y_2 = self.lambda_after
        lambda_t = [logistics_from_cps(t_i, x_1, y_1, x_2, y_2) for t_i in t]

        return lambda_t



if __name__ == '__main__':
    lambda_before = 0.5
    lambda_after = 0.2
    date_begin = datetime.datetime(2020,4,4)
    length = datetime.timedelta(days=1)

    cp = ChangePoint(lambda_before,lambda_after,date_begin,length,"sigmoid")
    a = cp.get_lambdas(0.001)

    import matplotlib.pyplot as plt
    plt.plot(a)