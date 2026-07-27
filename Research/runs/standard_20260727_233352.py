import numpy
import scipy


      import numpy as np
      from scipy.integrate import odeint

      # define the model
      def model(state, t):
         x, y, vx, vy = state
         ax = 0
         ay = -9.81  # gravity
         return [vx, vy, ax, ay]

      # initial condition
      state0 = [0, 0, 10, 0]

      # time points
      t = np.linspace(0, 10)

      # solve ODE
      state = odeint(model, state0, t)

      # print the result
      print(state)
   