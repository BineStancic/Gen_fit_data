#import libraries
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import matplotlib.ticker as tck

#fits class
class fits:

	#Construct everything thats needed
	def __init__(self, tau_1, tau_2, t_min, t_max, theta_min, theta_max, ratio, imax, ReallySmallNumber):
		self.tau_1 = tau_1
		self.tau_2 = tau_2
		self.t_min = t_min
		self.t_max = t_max
		self.theta_min = theta_min
		self.theta_max = theta_max
		self.ratio = ratio
		self.imax = imax
		self.ReallySmallNumber = ReallySmallNumber


	#Defining the normalisation constants
	def N_1(self):
		return(3.*np.pi*self.tau_1*(1.-np.exp(-10./self.tau_1)))

	def N_2(self):
		return(3.*np.pi*self.tau_2*(1.-np.exp(-10./self.tau_2)))


	#Defining the two components of the PDF
	def P_1(self, t, theta):
		return (1/fits.N_1(self) * (1 + (np.cos(theta)**2)) * np.exp(-t/self.tau_1))

	def P_2(self, t, theta):
		return (1/fits.N_2(self) * 3 *(np.sin(theta))**2 * np.exp(-t/self.tau_2))

	#Defining the function that combines the two components of the PDF
	def function(self, t ,theta):
		return (self.ratio*fits.P_1(self, t, theta) + (1-self.ratio)*fits.P_2(self, t, theta))

	##############################################################
	##############################################################


	#Box method that generates list of decay times and scattering angles
	def box_method(self):
		ymax = 0.215
		times_array = []
		theta_array = []

		t = [np.random.uniform(self.t_min, self.t_max) for _ in range(self.imax)]
		theta = [np.random.uniform(self.theta_min, self.theta_max) for _ in range(self.imax)]

		y1 = np.zeros(self.imax)
		y2 = np.zeros(self.imax)
		for i in range(self.imax):
			y1[i] = fits.function(self, t[i], theta[i])
			y2[i] = ymax*np.random.uniform()

		#box method
		i = 0
		while i < (self.imax):
			if (y2[i]<y1[i]):
				times_array.append(t[i])
				theta_array.append(theta[i])
				i = i + 1
			else:
				t[i] = np.random.uniform(0, self.t_max)
				theta[i] = np.random.uniform(0, self.theta_max)
				y1[i] = fits.function(self, t[i], theta[i])
				y2[i] = np.random.uniform()*ymax
				i == i
		return times_array, theta_array


	#Re define the normalisation constants and PDF with tau1 and tau2 being variables used for parts 2,3,4.
	def N_1min(self, tau_one):
		return(3.*np.pi*tau_one*(1.-np.exp(-10./tau_one)))

	def N_2min(self, tau_two):
		return(3.*np.pi*tau_two*(1.-np.exp(-10./tau_two)))

	def P_1min(self, t, theta, tau_one):
		return (1/fits.N_1min(self, tau_one) * (1 + (np.cos(theta)**2)) * np.exp(-t/tau_one))

	def P_2min(self, t, theta, tau_two):
		return (1/fits.N_2min(self, tau_two) * 3 *(np.sin(theta))**2 * np.exp(-t/tau_two))

	def functionmin(self, t ,theta, tau_one, tau_two, F):
		return (F*fits.P_1min(self, t, theta, tau_one) + (1-F)*fits.P_2min(self, t, theta, tau_two))


	#Minimiser from scipy optimize used for part 3 and part 4.
	def minimization_error(self, t_text, theta_text, initial_guess):
		#ReallySmallNumber = 1e-10
		#function that unpacks parameters and finds NLL
		def NLL_2(params):
			tau_one, tau_two, F = params
			joint_likelihood = -np.log(self.ReallySmallNumber+self.functionmin(t_text, theta_text, tau_one, tau_two, F))
			NLL = np.sum(joint_likelihood)
			return NLL
		#bounds over which the values can vary
		bounds = [(self.ReallySmallNumber, 10.), (self.ReallySmallNumber, 10.), (0, 1.)]
		#minimizer from scipy.optimize
		minimizer = minimize(NLL_2, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-10)
		min_NLL = minimizer.fun
		min_params = minimizer.x
		tau1_min = min_params[0]
		tau2_min = min_params[1]
		F_min = min_params[2]
		print('\n The minimum value of NLL is: ' +str(min_NLL))
		print('\n At the minumum NLL the estimated parameters tau1, tau2 and F are: \n' +str(min_params))

		##########################################################
		##########################################################

		#Error finding (not proper error)
		n_steps = 1000

		#generating linspace of x values and finding the y values of each.
		tau1_near_min = np.linspace(tau1_min-0.05,tau1_min+0.05,n_steps)
		tau2_near_min = np.linspace(tau2_min-0.05,tau2_min+0.05,n_steps)
		F_near_min = np.linspace(F_min-0.01,F_min+0.01,n_steps)

		NLL_near_min_tau1, NLL_near_min_tau2, NLL_near_min_F = [], [], []

		#Tau_1 ERROR
		for i in range(n_steps):
			NLL_near_min_tau1.append(NLL_2([tau1_near_min[i], tau2_min, F_min]))


		for i in range(0, n_steps):
			diff = NLL_near_min_tau1[i]-min_NLL
			loc_diff = tau1_near_min[i]
			error_tau1 = abs(loc_diff - tau1_min)
			if diff <= 0.5:
				break


		#Tau_2 ERROR
		for i in range(n_steps):
			NLL_near_min_tau2.append(NLL_2([tau1_min, tau2_near_min[i], F_min]))

		for i in range(0, n_steps):
			diff = NLL_near_min_tau2[i]-min_NLL
			loc_diff = tau2_near_min[i]
			error_tau2 = abs(loc_diff - tau2_min)
			if diff <= 0.5:
							break


		#F ERROR
		for i in range(n_steps):
			NLL_near_min_F.append(NLL_2([tau1_min, tau2_min, F_near_min[i]]))

		for i in range(0, n_steps):
			diff = NLL_near_min_F[i]-min_NLL
			loc_diff = F_near_min[i]
			error_F = abs(loc_diff - F_min)
			if diff <= 0.5:
							break
		err_array = [error_tau1, error_tau2, error_F]
		print('\n The errors calculated for tau1, tau2 and F are: \n' +str(err_array))

		return(tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F)




	#finding the proper error by varying over tau 1 and minimising again for tau 2 and F
	def propah(self, t_text, theta_text, tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min, NLL_near_min_F):
		
		#Define the range and the position over which to look for the true minimum NLL
		n_steps = 100
		cut = 450

		#TAU 1
		print('\n Proper error on tau1 calculation.')
		nll_array = []
		i = 0
		for i in range(n_steps):
			tau_one = tau1_near_min[cut+i]

			#function that unpacks parameters and finds NLL
			def NLL_2(params):
				tau_two, F = params
				joint_likelihood = -np.log(self.ReallySmallNumber+self.functionmin(t_text, theta_text, tau_one, tau_two, F))
				NLL = np.sum(joint_likelihood)
				return NLL

			#bounds over which the values can vary
			bounds = [(self.ReallySmallNumber, 10.), (0, 1.)]
			initial_guess = [2., 0.5]

			#minimizer from scipy.optimize and append the results to arrays
			minimizer = minimize(NLL_2, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-10)	
			min_NLL = minimizer.fun
			nll_array.append(min_NLL)
		
		#Find the index at which the NLL is minimum and find the corresponding tau 1 value 
		argg = np.argmin(nll_array)
		min_NLL = nll_array[argg]
		print('The new minimum value of NLL is: ' +str(min_NLL))
		tau_1_min = tau1_near_min[cut+argg]


		#Find the error for minimum NLL
		for i in range(0, n_steps):
			diff = nll_array[i]-min_NLL
			loc_diff = tau1_near_min[cut+i]
			error_tau1 = abs(loc_diff - tau_1_min)
			if diff <= 0.5:
				break




		#Tau 2
		print('\n Proper error on tau2 calculation.')
		nll_array = []
		i = 0
		for i in range(n_steps):
			tau_two = tau2_near_min[cut+i]

			#function that unpacks parameters and finds NLL
			def NLL_2(params):
				tau_one, F = params
				joint_likelihood = -np.log(self.ReallySmallNumber+self.functionmin(t_text, theta_text, tau_one, tau_two, F))
				NLL = np.sum(joint_likelihood)
				return NLL

			#bounds over which the values can vary
			bounds = [(self.ReallySmallNumber, 10.), (0, 1.)]
			initial_guess = [1., 0.5]

			#minimizer from scipy.optimize
			minimizer = minimize(NLL_2, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-10)
			min_NLL = minimizer.fun
			nll_array.append(min_NLL)
				
		#Find the index at which the NLL is minimum and find the corresponding tau 2 value 
		argg = np.argmin(nll_array)
		min_NLL = nll_array[argg]
		print('The new minimum value of the NLL is: ' +str(min_NLL))
		tau_2_min = tau2_near_min[cut+argg]

		#Find error for minimum NLL
		for i in range(0, n_steps):
			diff = nll_array[i]-min_NLL
			loc_diff = tau2_near_min[cut+i]
			error_tau2 = abs(loc_diff - tau_2_min)
			if diff <= 0.5:
				break


		#F
		print('\n Proper error on F calculation')
		nll_array = []
		i = 0
		for i in range(n_steps):
			F = F_near_min[cut+i]

			#function that unpacks parameters and finds NLL
			def NLL_2(params):
				tau_one, tau_two = params
				joint_likelihood = -np.log(self.ReallySmallNumber+self.functionmin(t_text, theta_text, tau_one, tau_two, F))
				NLL = np.sum(joint_likelihood)
				return NLL

			#bounds over which the values can vary
			bounds = [(self.ReallySmallNumber, 10.), (self.ReallySmallNumber, 10.)]
			initial_guess = [1., 2.]

			#minimizer from scipy.optimize
			minimizer = minimize(NLL_2, initial_guess, method='L-BFGS-B', bounds=bounds, tol=1e-10)
			min_NLL = minimizer.fun
			nll_array.append(min_NLL)

		#Find the index at which the NLL is minimum and find the corresponding F value 
		argg = np.argmin(nll_array)
		min_NLL = nll_array[argg]
		print('The new minimised NLL is: ' +str(min_NLL))
		F_min = F_near_min[cut+argg]

		for i in range(0, n_steps):
			diff = nll_array[i]-min_NLL
			loc_diff = F_near_min[cut+i]
			error_F = abs(loc_diff - F_min)
			if diff <= 0.5:
				break

		#Print out all errors
		print('Tau1 min: ' +str(tau_1_min))
		print('Proper error on Tau1:  ' +str(error_tau1))

		print('tau2 min ; ' +str(tau_2_min))
		print('Proper error on Tau2:  ' +str(error_tau2))

		print('F min: ' +str(F_min))
		print('Proper error on F:  ' +str(error_F))


	def plotting_simulated(self, times_array, theta_array):
		nbins = int(np.sqrt(self.imax))

		#plots of euler and rk4 on subplots
		plt.hist(times_array, bins = nbins)
		plt.xlabel('time')
		plt.ylabel('frequency')
		plt.xlim(self.t_min, self.t_max)
		#plt.savefig('times.png', dpi=360)
		plt.show()

		plt.hist(theta_array, bins = nbins)
		plt.xlabel('theta (radians)')
		plt.ylabel('frequency')
		plt.xlim(self.theta_min, self.theta_max)
		#plt.savefig('theta.png', dpi=360)
		plt.show()


	def plotting_NLL(self, tau1_near_min, NLL_near_min_tau1, tau2_near_min, NLL_near_min_tau2, F_near_min,\
					 NLL_near_min_F, tau1_min, error_tau1, tau2_min,error_tau2, F_min, error_F):
		#Plot of tau1 against NLL with errors
		plt.plot(tau1_near_min, NLL_near_min_tau1, color='k')
		leftedge = tau1_min-error_tau1
		rightedge = tau1_min+error_tau1
		plt.axvspan(leftedge, rightedge, color='lightcoral', label='error on tau1')
		plt.axvline(x=tau1_min, color='b', label='tau1 min')
		plt.xlabel('tau1')
		plt.ylabel('NLL')
		plt.legend()
		plt.savefig('3tau1_error.png', dpi=360)
		plt.show()

		#Plot of tau2 against NLL with errors
		plt.plot(tau2_near_min, NLL_near_min_tau2, color='k')
		leftedge = tau2_min-error_tau2
		rightedge = tau2_min+error_tau2
		plt.axvspan(leftedge, rightedge, color='lightcoral', label='error on tau 2')
		plt.axvline(x=tau2_min, color='b', label='tau2 min')
		plt.xlabel('tau2')
		plt.ylabel('NLL')
		plt.legend()
		plt.savefig('3tau2_error.png', dpi=360)
		plt.show()


		#Plot of F against NLL with errors
		plt.plot(F_near_min, NLL_near_min_F, color='k')
		leftedge = F_min-error_F
		rightedge = F_min+error_F
		plt.axvspan(leftedge, rightedge, color='lightcoral', label='error F')
		plt.axvline(x=F_min, color='b', label='F min')
		plt.xlabel('F')
		plt.ylabel('NLL')
		plt.legend()
		plt.savefig('3F_error.png', dpi=360)
		plt.show()