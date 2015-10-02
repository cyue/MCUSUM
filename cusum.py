import sys
import numpy as np
from scipy.stats import norm

class CUSUM:
    
    k = None
    h = None
    delta = None

    mu = None
    sigma_inv = None

    # number of sample examined 
    cnt = 0 

    # positive multivariate cumulative sum
    _pmc = 0
    # negative multivariate cumulative sum
    _nmc = 0

    def __init__(self, delta=1, h=None):
        ''' @k is the reference value
            @h is the control limit
        '''
        self.k = None
        self.h = h
        self.delta = delta
        self.mu = None
        self.sigma_inv = None
        self._pmc = 0
        self._nmc = 0

        self.cnt = 0


    def read(self, sample):
        ''' read '''
        self.cnt = self.cnt + 1
        return np.asarray(sample)

    
    def update_params(self, sample):
        ''' @mu and @sigma_inv are iterative calculation
            k = 0.5 * sqrt((mu1 - mu0)'Sigma(mu1 - mu0)) '''
        mu = self.mean(sample)
        sigma_inv = self.std_inv(sample)
        self.mu = mu
        self.sigma_inv = sigma_inv

        dim = sigma_inv.shape[0]
        # simplify calculation by transform
        sigma_diag = np.sqrt(np.diag(np.linalg.inv(sigma_inv)).reshape(dim,1))
        self.k = dim + 0.5 * np.dot(np.dot(self.delta * sigma_diag.T,
                                        self.sigma_inv), 
                    self.delta * sigma_diag)
        

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


    def positive_cusum(self, sample):
        ''' '''
        # set initial cusum 0 
        if self.cnt <= 1:
            return 0
        # reshape sample and mu to simplify calculation
        sample = sample.reshape(len(sample),1)
        mu = self.mu.reshape(len(self.mu),1)

        d_square = np.dot(np.dot((sample - mu).T, self.sigma_inv),
                        (sample - mu))
        return max(0, (self._pmc + d_square - self.k)[0][0])


    def negative_cusum(self, sample):
        ''' '''
        # set initial cusum 0 
        if self.cnt <= 1:
            return 0
        # reshape sample and mu to simplify calculation
        sample = sample.reshape(len(sample),1)
        mu = self.mu.reshape(len(self.mu),1)

        d_square = -1*np.dot(np.dot((sample - mu).T, self.sigma_inv),
                        (sample - mu))
        return min(0, (self._nmc + d_square + self.k)[0][0])
               
          
    def predict(self, sample):
        ''' '''
        sample = self.read(sample)

        prediction = 0
        self._pmc = self.positive_cusum(sample)
        self._nmc = self.negative_cusum(sample)
        if self._pmc > self.h or self._nmc < -self.h:
            prediction = 1  
        self.update_params(sample)

        return prediction

def main():
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    model = CUSUM(delta=3, h=2)
    for exmp in data[:,:2]:
        pred = model.predict(exmp)
        print pred, model._pmc, model._nmc, model.k[0][0]


def test():
    data = np.genfromtxt(sys.argv[1], delimiter=',')
    for limit in xrange(1,3000, 100):
        #out = open(str(limit)+'.d', 'w')
        model = CUSUM(delta=1, h=limit*1)

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
