# Gen_fit_data
#Generating and fitting data of a 2 dimensional PDF

#generating_fitting contains the fits class which is then imported into the main code.

#datafile-Xdecay.txt is a text file with values of t [:, 0] and theta [:, 1]

#The PDF is a 2 dimensional PDF describing the decay of a particle to its daughter particle at time t and at angle theta. The PDF takes
#in three parameters tau1, tau2 and F.

#In first part of the code the parameters are inputed to generate obserable data of time and theta using a monte carlo box method.

#In the second part of the code values of times are read from datafile-Xdecay.txt and theta is chosen to be a random number in its range.
#A negative log likelihood funcion is defined for the PDF and scipy.optimize minimiser is used to find the minimum values, hence the best
#fit for the parameters tau1, tau2 and F. The errors are found using a simple method, by fixing two parameters and varying the third around
#the minimum NLL value. The error being the value of the parameter at NLL minimum + 0.5.

#This is then repeated using the full data of time and theta.

#Lastly the proper errors were found for parts 2 and 3. MIUNUIT could have been used to found these proper errors, but this is a manual 
#way to do it. For each parameter varying around its minimum value the other two values were re minimised in order to find the true
#minimum NLL value. The proper error being the value of the parameter at  NLL minimum + 0.5. This was repeated for the remainding
#two parameters to find their proper errors as well.
