import numpy as np
import pandas as pd
import math
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
np.random.seed(seed=11)
#Initializing default values for global variables


class ExpectationMaximizationModel:

    def InitializeModel(self,file,cluster):
        self.data = np.genfromtxt(file, delimiter=',') #Data points
        self.threshold = 0.01  # for stopping EM
        self.k = cluster     # Number of clusters
        self.n = self.data.shape[0]   # Number of points
        self.d = self.data.shape[1]  # Number of dimensions
        self.current_log_ll=0     #Current log likelihood
        self.loops_iteration=100 #Iterations of EM for each k
        self.random_restart=10 # Number of random restarts
        self.random=0
        self.BIC=0
        self.best=0

       #Initializing the Mus(means) and covariance(Sigma) for each gaussian cluster

        self.Mus = self.data[np.random.choice(self.n, self.k, False), :]
        var = (np.var(self.data)) / self.k
        self.Sigma = [np.eye(self.d)] * self.k
        for i in range(0,len(self.Sigma)):
            self.Sigma[i]=self.Sigma[i]*var

        #print(self.Sigma)
        #print(self.Mus)

        #Priors Initialization for each of the clusters P(x_point|Cluster)
        self.priors = [1. / self.k] * self.k

        #Responsibility Matrix for the model to calculate probablity P(Cluster|x_point)
        self.Resp = np.zeros((self.n, self.k))

        #Lists needed to store the random restart states
        self.log_ll_list = []
        self.best_log_ll = []
        self.best_mus = []
        self.best_sigma = []
        self.best_Resp = []
        self.best_log_ll_for_iter = []

    def ExpectationMaximization(self):

        while len(self.log_ll_list) < self.loops_iteration:

            #EXPECTATION : where all the probablities are calculated
            for cluster in range(self.k):
                #STEP 1
                # Calculating using the multivariate probablity distribution equation
                self.Resp[:, cluster] = self.priors[cluster] * multivariate_normal.pdf(self.data, self.Mus[cluster], self.Sigma[cluster])
                # print("After multiplyin")
                # print(R)

            #STEP 2
            #Calculating log likelihood
            self.current_log_ll = np.sum(np.log(np.sum(self.Resp, axis=1)))

            #print("LOG-LIKELIHOOD")
            #print(log_likelihood)
            #print("Iteration")
            #print(len(log_likelihoods))
            #lk = np.sum(np.sum(R, axis=1))

            self.log_ll_list.append(self.current_log_ll)
            #print(self.log_ll_list)

            #Calculating the Posterior probablities
            num=self.Resp.T
            denom=np.sum(self.Resp, axis=1)
            self.Resp = (num/denom).T

            #MAXIMIZATION : Here we update the Mus and covariances of the gaussians

            #STEP1
            #Now we calculate the sum of the probabilities for each cluster from all points to calculate new Mus and Sigmas
            denom = np.sum(self.Resp, axis=0)

            #STEP2
            #Changing Mus,Sigmas
            for k in range(self.k):

                self.Mus[k] = 1. / denom[k] * np.sum(self.Resp[:, k] * self.data.T, axis=1).T
                point_mu = np.matrix(self.data - self.Mus[k])
                #print(self.Mus[k])
                self.Sigma[k] = np.array(1 / denom[k] * np.dot(np.multiply(point_mu.T, self.Resp[:, k]), point_mu))

                self.priors[k] = 1. / self.n * denom[k]

            if len(self.log_ll_list) < 2: continue
            if ((len(self.log_ll_list) == self.loops_iteration) or (np.abs(self.current_log_ll - self.log_ll_list[-2]) < self.threshold )):
                #print("Reached End of Maximization , Might go to restart")

                #RANDOM RESTART
                #Storing the states before randomly restarting
                self.best_log_ll.append(self.current_log_ll)
                self.best_mus.append(self.Mus)
                self.best_sigma.append(self.Sigma)
                self.best_Resp.append(self.Resp)
                self.best_log_ll_for_iter.append(self.log_ll_list)

                #print("Log Likelihood")
                #print(log_likelihood)
                # append mus,sigmas
                if self.random > self.random_restart:
                    #print("Done with restarts")
                    best = np.argmax(self.best_log_ll)

                    sum = 0
                    self.Mus = self.best_mus[best]
                    self.Sigma = self.best_sigma[best]
                    self.Resp = self.best_Resp[best]
                    self.log_ll_list = self.best_log_ll_for_iter[best]

                    param = (self.k - 1) + (self.k * self.d) + (self.d * self.d)

                    self.BIC = (param * math.log(self.n)) - (2 * (self.best_log_ll[best]))

                    return
                #print("RANDOM RESTARTING")
                #print("iter_count")
                #print(len(self.best_log_ll))

                #RESETTING ALL THE MODEL PARAMETERS FOR RANDOM RESTART
                self.log_ll_list  = []
                self.Mus = self.data[np.random.choice(self.n, self.k, False), :]

                var = (np.var(self.data)) / self.k
                # initialize the covariance matrices for each gaussians
                self.Sigma = [np.eye(self.d)] * self.k

                for i in range(0, len(self.Sigma)):
                    self.Sigma[i] = self.Sigma[i] * var

                self.priors = [1. / self.k] * self.k
                self.Resp = np.zeros((self.n, self.k))
                self.random += 1
                #print(self.random)







    def print_result_part1(self):
        best = np.argmax(self.best_log_ll)

        sum = 0
        self.Mus = self.best_mus[best]
        self.Sigma = self.best_sigma[best]
        self.Resp = self.best_Resp[best]
        self.log_ll_list = self.best_log_ll_for_iter[best]

        param = (self.k - 1) + (self.k * self.d) + (self.d * self.d)

        self.BIC = (param * math.log(self.n)) - (2 * (self.best_log_ll[best]))
        # print("BIC")
        # print(BIC)
        print("The best cluster centers:")
        for i in range(0,len(self.Mus)):
            print("Cluster:",(i+1))
            print(self.Mus[i,:])
        #print(self.Mus)
        print("The covariances of the best gaussian clusters:")


        for i in range(0,len(self.Sigma)):
            print("Covariance:",(i+1))
            print(self.Sigma[i])
        #print(self.Sigma)

        #print("Log Likelihood list")
        #print(self.log_ll_list)

        print("Log likelihood of the model(Max)")
        print(self.best_log_ll[self.best])




def find_best_number_of_clusters(dat):
    BIC_list=[]

    points = np.genfromtxt(dat, delimiter=',')
    n, d = points.shape
    print("Finding best number of clusters")
    for c in range(2,n):
        EM_model=ExpectationMaximizationModel()
        EM_model.InitializeModel(dat, c)
        EM_model.ExpectationMaximization()
        BIC=EM_model.BIC
        BIC_list.append(BIC)

        print("BIC for number of clusters ",c," is ",BIC)
        if(len(BIC_list)>=2):
            if(BIC>BIC_list[-2]):
                print("Since the BIC is increasing now, we stop.")
                print("The best number of clusters = ",c-1)


                return c-1
                break



def sketch_log_vs_iteration(b_c):
    print("QUESTION 6:")
    print("Plotting Likelihood Vs Iteration for best number of cluster from previous step")
    print("Number of clusters=",b_c)
    EM_model = ExpectationMaximizationModel()
    EM_model.InitializeModel('test_points.csv', b_c)
    print("Previous Threshold Difference :",EM_model.threshold)
    EM_model.ExpectationMaximization()
    log_like = EM_model.best_log_ll_for_iter[model.best]
    plt.plot(log_like)
    plt.title('Log Likelihood vs iteration plot for old threshold')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    print("The best cluster centers:")
    for i in range(0, len(EM_model.Mus)):
        print("Cluster:", (i + 1))
        print(EM_model.Mus[i, :])
    # print(self.Mus)
    print("The covariances of the best gaussian clusters:")

    for i in range(0, len(EM_model.Sigma)):
        print("Covariance:", (i + 1))
        print(EM_model.Sigma[i])
    # print(self.Sigma)

    # print("Log Likelihood list")
    # print(self.log_ll_list)

    print("Log likelihood of the model(Max)")
    print(EM_model.best_log_ll[EM_model.best])
    #
    # Setting new Threshold
    EM_model = ExpectationMaximizationModel()
    EM_model.InitializeModel('test_points.csv', b_c)
    EM_model.threshold = 0.0001

    print("Setting new Threshold value to much lower, which is = ", EM_model.threshold)
    EM_model.ExpectationMaximization()
    log_like = EM_model.best_log_ll_for_iter[model.best]
    print("Plotting now")
    plt.plot(log_like)
    plt.title('Log Likelihood vs iteration plot for new threshold')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()
    print("The best cluster centers:")
    for i in range(0, len(EM_model.Mus)):
        print("Cluster:", (i + 1))
        print(EM_model.Mus[i, :])
    # print(self.Mus)
    print("The covariances of the best gaussian clusters:")

    for i in range(0, len(EM_model.Sigma)):
        print("Covariance:", (i + 1))
        print(EM_model.Sigma[i])
    # print(self.Sigma)

    # print("Log Likelihood list")
    # print(self.log_ll_list)

    print("Log likelihood of the model(Max)")
    print(EM_model.best_log_ll[EM_model.best])



if __name__ == "__main__":

    data_file=str(input("Enter file name"))
    cluster_num=int(input("Enter number of clusters"))
    print("STARTING")
    model=ExpectationMaximizationModel()
    model.InitializeModel(data_file,cluster_num)
    model.ExpectationMaximization()
    model.print_result_part1()
    b_c=find_best_number_of_clusters(data_file)
    sketch_log_vs_iteration(b_c)


