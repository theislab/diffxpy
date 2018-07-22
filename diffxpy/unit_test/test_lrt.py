import os
import unittest
import tempfile

import numpy as np
import xarray as xr
import pandas as pd

from sgdglm.api.models.nb_glm import Simulator, Estimator, InputData
import diffxpy as de


# from utils.config import getConfig


class TestLRT(unittest.TestCase):
    sim: Simulator
    working_dir: tempfile.TemporaryDirectory
    
    def setUp(self):
        self.sim = Simulator(num_observations=2000, num_features=100)
        self.sim.generate()
        self.working_dir = tempfile.TemporaryDirectory()
        
        print("working_dir: %s" % self.working_dir)
    
    # def tearDown(self):
    #     for e in self._estims:
    #         e.close_session()
    #
    #     self.working_dir.cleanup()
    
    def test_lrt(self):
        sim = self.sim
        wd = os.path.join(self.working_dir.name, "lrt")
        os.makedirs(wd, exist_ok=True)
        
        test = de.test_lrt(sim.input_data, full_formula="~ 1 + condition", reduced_formula="~ 1")
        
        print(test.summary())


if __name__ == '__main__':
    unittest.main()
