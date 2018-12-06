#import libraries
import numpy as np
from generating_fitting import *


#################################################################

#defining parameters
ratio = 0.5
tau_1 = 1.
tau_2 = 2.
t_min = 0
t_max = 10
theta_min = 0
theta_max = 2*np.pi
imax = 10000
ReallySmallNumber = 1e-10

#importing class
fits = fits(tau_1, tau_2, t_min, t_max, theta_min, theta_max, ratio, imax, ReallySmallNumber)


##################################################################
#Part 1
decayt, decaytheta = fits.box_method()
fits.plotting_simulated(decayt, decaytheta)

#reading in the .txt and appending values to two arrays
textyboi = np.loadtxt(open('datafile-Xdecay.txt',"r"))
t_text = textyboi[:, 0]
theta_text = textyboi[:, 1]

##################################################################
#Part 2 and corresponding 4
def main():
    theta_rand = np.random.uniform(theta_min, theta_max, 100000)
    initial_guess = [1., 2., 0.5]
    tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F=fits.minimization_error(t_text, theta_rand, initial_guess)

    fits.plotting_NLL(tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F)

    fits.propah(t_text, theta_text, tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F)

#main()

##################################################################
#Part 3 and corresponding 4
def main2():
    initial_guess = [1., 2., 0.5]
    tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F=fits.minimization_error(t_text, theta_text, initial_guess)

    fits.plotting_NLL(tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F)

    fits.propah(t_text, theta_text, tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F)
main2()