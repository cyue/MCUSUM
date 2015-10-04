import numpy as np
import time
from v2cusum import *
from generate_synthetic import *

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


def eff_test():
    ''' efficiency test wrt prediction time cost '''
    size = 1000
    mu = np.array([5,5])
    cov_matrix = np.array([[1,0.1],[0.1,1]])
    for coef in xrange(1,11):
        mus, cov_matrices = create_params(mu,
                            cov_matrix,
                            0.1,
                            [[0.,0.],[0.,0.]],
                            coef*size)
        samples = generate(mus, cov_matrices, 1)
        model = CUSUM(delta=2, h=0.1)
        start = time.time()
        for sample in samples:
            model.predict(sample)
        cost = time.time() - start
        print coef*size, cost, ';' 


if __name__ == '__main__':
    #test()
    eff_test()
