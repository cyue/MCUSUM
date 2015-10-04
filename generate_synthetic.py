import sys
import numpy as np

def generate(mus, cov_matrices, batch_size=10):
    ''' '''
    group_size = len(mus)
    dim = cov_matrices.shape[1]

    samples = []
    for i in xrange(group_size):
        sample = np.random.multivariate_normal(mus[i], 
                                                cov_matrices[i], 
                                                batch_size)
        # add label
        sample = [np.append(item,0) for item in sample]
        # add normal sample with label
        samples = np.append(samples, sample)
        # noise level = 5, noise magnitude = 2
        # add noise with label
        for point in sample:
            if np.random.rand()*100 <= 5:
                noise = point[:2] + np.sqrt(np.diag(cov_matrices[i]))
                samples = np.append(samples, np.append(noise, 1))      
    # add label dimension
    samples = samples.reshape(len(samples)/(dim+1), dim+1)
    
    return samples


def create_params(mu, cov_matrix, mean_shift, cov_shift, size):
    ''' mean_shift is scalar, shift same size for all dimension
        cov_shift is symmetric matrix '''
    mus = []
    cov_matrices = []
    for i in xrange(size):
        mus.append(mu)
        mu = mu + mean_shift

        cov_matrices.append(cov_matrix)
        cov_matrix = cov_matrix + cov_shift

    return np.asarray(mus), np.asarray(cov_matrices)
        

def main():
    ''' '''
    initial_mu = np.array([5,5])
    initial_cov_matrix = np.array([[1,0.1],[0.1,1]])

    mus, cov_matrices = create_params(initial_mu,
                            initial_cov_matrix,
                            0.5,
                            [[0.,0.01],[0.,0.01]],
                            100)
    samples = generate(mus, cov_matrices, 10)
    for sample in samples:
        print ','.join([str(item) for item in sample])


if __name__ == '__main__':
    main()

