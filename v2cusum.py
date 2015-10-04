import sys
import numpy as np
from scipy.stats import norm

class CUSUM:
    
    h = None
    delta = None

    mu = None
    sigma_inv = None

    # number of sample examined 
    cnt = 0 

    # positive multivariate cumulative sum
    _cusum = 0

    def __init__(self, delta=1, h=None):
        ''' 
            @h is the control limit
        '''
        self.h = h
        self.delta = delta
        self.mu = None
        self.sigma_inv = None
        self._cusum = 0

        self.cnt = 0


    def read(self, sample):
        ''' read new sample'''
        self.cnt = self.cnt + 1
        return np.asarray(sample)

    
    def updates(self, sample):
        ''' @mu and @sigma_inv are iterative calculation
        '''
        mu = self.mean(sample)
        sigma_inv = self.std_inv(sample)
        self.mu = mu
        self.sigma_inv = sigma_inv


    def mean(self, sample):
        ''' iterative calculation of mean '''
        # mu for first example
        if self.mu is None:
            return sample

        mu = self.mu + 1 / self.cnt * (sample - self.mu)
        return mu

    def std_inv(self, sample):
        ''' iteratively calcuate inverse of sigma '''
        # manually set std_inv for first two example
        if self.cnt <= 2:
            # define the shape of inverse of covariance matrix
            inv_cov_shape = (len(sample), len(sample))
            return np.linalg.inv(100 * np.eye(len(sample)) + 
                    np.zeros(inv_cov_shape))

        # reshape sample and mu to simplify calculation
        mu = self.mu.reshape(len(self.mu), 1)
        sample = sample.reshape(len(sample), 1)
        
        cnt = self.cnt - 1
        dim = mu.shape[0]
        new_sigma_inv = np.dot(cnt * self.sigma_inv / (cnt-1),
            np.eye(dim) - np.dot(np.dot(sample - mu, (sample - mu).T),
                                self.sigma_inv) / \
                ((cnt**2-1)/cnt + np.dot(np.dot((sample - mu).T, self.sigma_inv),
                                (sample - mu))) )
                
        return new_sigma_inv


    # no scale
    def max_cusum(self, sample):
        ''' calculate mac cusum based on 
            S(n) = max{0, S(n-1) + log_ratio)'''
        pass
        '''
        # set initial cusum 0 
        if self.cnt <= 1:
            return 0
        # reshape sample and mu to simplify calculation
        sample_vector = sample.reshape(len(sample),1)
        mu_vector = self.mu.reshape(len(self.mu),1)

        dim = sigma_inv.shape[0]
        sigma_vector = np.sqrt(np.diag(np.linalg.inv(sigma_inv)).reshape(dim,1))

        #TO DO HERE
        log_ratio = self.delta * np.dot(np.dot(sigma_vector.T, self.sigma_inv), 
                                    sample_vector) - 
                    0.5 * np.dot(np.dot((2*mu_vector + self.delta*sigma_vector).T, 
                                        self.sigma_inv), 
                                 self.delta*sigma_vector)

        return max(0, (self._cusum + log_ratio)[0][0])
        '''

    # std scale
    def max_cusum(self, sample, scale=True):
        ''' calculate max cusum based on 
            C(n) = max{0, C(n-1) + Zi - 0.5k) 
            where Zi = alpha(sample - mu),
            k = sqrt(new_mu - mu)'Sigma_inv(new-mu - mu)
        '''
        # set initial cusum 0 
        if self.cnt <= 1:
            return 0
        # reshape sample and mu to simplify calculation
        sample_vector = sample.reshape(len(sample),1)
        mu_vector = self.mu.reshape(len(self.mu),1)

        dim = self.sigma_inv.shape[0]
        sigma_vector = np.sqrt(np.diag(np.linalg.inv(self.sigma_inv)).reshape(dim,1))

        #TO DO HERE
        k = self.delta * np.sqrt(
                            np.dot(
                                np.dot(sigma_vector.T, self.sigma_inv),
                                sigma_vector)
                                )
        alpha = self.delta * np.dot(sigma_vector.T, self.sigma_inv) / k
        zeta = alpha * (sample_vector - mu_vector)

        return max(0, (self._cusum + zeta - 0.5*k)[0][0])

          
    def predict(self, sample):
        ''' '''
        sample = self.read(sample)

        prediction = 0
        self._cusum = self.max_cusum(sample,scale=True)
        if np.fabs(self._cusum) > self.h: 
            prediction = 1  
        self.updates(sample)

        return prediction

def main():
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    model = CUSUM(delta=1.5, h=2)
    for exmp in data[:,:2]:
        pred = model.predict(exmp)
        print pred, model._pmc, model._nmc, model.k[0][0]


def test():
    ''' input is 2D synthetic data '''
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    for limit in xrange(1,100):
        #out = open(str(limit)+'.d', 'w')
        model = CUSUM(delta=2, h=limit*0.01)

        tp,fp,tn,fn = 0,0,0,0
        for idx, exmp in enumerate(data[:,:2]):
            pred = model.predict(exmp)
            tp = tp + (1 if pred == data[idx][2] and pred == 1 else 0)
            fp = fp + (1 if (pred - data[idx][2]) == 1 else 0)
            tn = tn + (1 if pred == data[idx][2] and pred == 0 else 0)
            fn = fn + (1 if (data[idx][2] - pred) == 1 else 0)
        print np.float(tp)/(tp+fn), np.float(fp)/(fp+tn)


if __name__ == '__main__':
    test()
