#!/usr/bin/env python
"""Toolbox for Weighted MinHash Algorithms

This module contains the following algorithms: the standard MinHash algorithm for binary sets and
weighted MinHash algorithms for weighted sets. 
Each algorithm converts a data instance (i.e., vector) into the hash code of the specified length, 
and computes the time of encoding.

Authors
-----------
Wei WU

Usage
---------
    >>> from WeightedMinHash import WeightedMinHash
    >>> wmh = WeightedMinHash.WeightedMinHash(data, dimension_num, seed)
    >>> fingerprints_k, fingerprints_y, elapsed = wmh.algorithm_name(...)
      or
    >>> fingerprints, elapsed = wmh.algorithm_name(...)

Parameters
----------
data: {array-like, sparse matrix}, shape (n_features, n_instances)
    a data matrix where row represents feature and column is data instance

dimension_num: int
    the length of hash code

seed: int, default: 1
    part of the seed of the random number generator

Returns
-----------
fingerprints_k: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance

fingerprints_y: ndarray, shape (n_instances, dimension_num)
    one component of hash code from some algorithms, and each row is the hash code for a data instance

fingerprints: ndarray, shape (n_instances, dimension_num)
    only one component of hash code from some algorithms, and each row is the hash code for a data instance

elapsed: float
    time of hashing data matrix

Note
-----------
"""

import numpy as np
import numpy.matlib
import scipy as sp
import scipy.sparse as sparse
import time
from ctypes import *



class WeightedMinHash:
    """Main class contains 13 algorithms

    Attributes:
    -----------
    PRIME_RANGE: int
        the range of prime numbers

    PRIMES: ndarray
        a 1-d array to save all prime numbers within PRIME_RANGE, which is used to produce hash functions
                 $\pi = (a*i+b) mod c$, $a, b, c \in PRIMES$
        The two constants are used for minhash(self), haveliwala(self, scale), haeupler(self, scale)

    weighted_set: {array-like, sparse matrix}, shape (n_features, n_instances)
        a data matrix where row represents feature and column is data instance

    dimension_num: int
        the length of hash code

    seed: int, default: 1
        part of the seed of the random number generator. Note that the random seed consists of seed and repeat.

    instance_num: int
        the number of data instances

    feature_num: int
        the number of features
    """

    C_PRIME = 10000000000037

    def __init__(self, weighted_set, dimension_num, seed=1):

        self.weighted_set = weighted_set
        self.dimension_num = dimension_num
        self.seed = seed
        self.instance_num = self.weighted_set.shape[1]
        self.feature_num = self.weighted_set.shape[0]

    def minhash(self, repeat=1):
        """The standard MinHash algorithm for binary sets
           A. Z. Broder, M. Charikar, A. M. Frieze, and M. Mitzenmacher, "Min-wise Independent Permutations",
           in STOC, 1998, pp. 518-529

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ---------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            k_hash = np.mod(
                np.dot(np.transpose(np.array([feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def haveliwala(self, repeat=1, scale=1000):
        """[haveliwala et. al., 2000] directly rounds off the remaining float part
            after each weight is multiplied by a large constant.
            T. H. Haveliwala, A. Gionis, and P. Indyk, "Scalable Techniques for Clustering the Web",
            in WebDB, 2000, pp. 129-134

        Parameters
        ----------
        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operation of expanding the original weighted set by scaling the weights is implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        expanded_set_predefined_size = np.ceil(np.max(np.sum(self.weighted_set * scale, axis=0)) * 100).astype(int)
        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):
            expanded_feature_id = np.zeros((1, expanded_set_predefined_size))
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            expanded_set = CDLL('./cpluspluslib/haveliwala_expandset.so')
            expanded_set.GenerateExpandedSet.argtypes = [c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         c_int, c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS")]
            expanded_set.GenerateExpandedSet.restype = None
            feature_weight = np.round(np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0])
            expanded_feature_id = expanded_feature_id[0, :]

            expanded_set.GenerateExpandedSet(expanded_set_predefined_size, feature_weight, feature_id,
                                             feature_id_num, scale, expanded_feature_id)

            expanded_feature_id = expanded_feature_id[expanded_feature_id != 0]
            expanded_feature_id_num = expanded_feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([expanded_feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((expanded_feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)
            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = expanded_feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def haeupler(self, repeat=1, scale=1000):
        """[Haeupler et. al., 2014] preserves the remaining float part with probability
           after each weight is multiplied by a large constant.
           B. Haeupler, M. Manasse, and K. Talwar, "Consistent Weighted Sampling Made Fast, Small, and Easy",
           arXiv preprint arXiv: 1410.4266, 2014

        Parameters
        ----------
        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operation of expanding the original weighted set by scaling the weights is implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        expanded_set_predefined_size = np.ceil(np.max(np.sum(self.weighted_set * scale, axis=0)) * 100).astype(int)
        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))

        for j_sample in range(0, self.instance_num):

            expanded_feature_id = np.zeros((1, expanded_set_predefined_size))
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            expanded_set = CDLL('./cpluspluslib/haeupler_expandset.so')
            expanded_set.GenerateExpandedSet.argtypes = [c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                flags="C_CONTIGUOUS"),
                                                         c_int, c_int, c_int,
                                                         np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                flags="C_CONTIGUOUS")]
            expanded_set.GenerateExpandedSet.restype = None
            feature_weight = np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0]
            expanded_feature_id = expanded_feature_id[0, :]
            expanded_set.GenerateExpandedSet(expanded_set_predefined_size, feature_weight, feature_id, feature_id_num,
                                             scale, self.seed * repeat, expanded_feature_id)

            expanded_feature_id = expanded_feature_id[expanded_feature_id != 0]
            expanded_feature_id_num = expanded_feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([expanded_feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((expanded_feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)
            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = expanded_feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def gollapudi1(self, repeat=1, scale=1000):
        """[Gollapudi et. al., 2006](1) is an integer weighted MinHash algorithm,
           which skips much unnecessary hash value computation by employing the idea of "active index".
           S. Gollapudi and R. Panigraphy, "Exploiting Asymmetry in Hierarchical Topic Extraction",
           in CIKM, 2006, pp. 475-482.

        Parameters
        -----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        scale: int, default: 1000
            a large constant to transform real-valued weights into integer ones

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operations of seeking "active indices" and computing hashing values are implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """
        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        for j_sample in range(0, self.instance_num):

            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            fingerprints = CDLL('./cpluspluslib/gollapudi1_fingerprints.so')
            fingerprints.GenerateFingerprintOfInstance.argtypes = [c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   c_int, c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS")]
            fingerprints.GenerateFingerprintOfInstance.restype = None
            feature_weight = np.array(scale * self.weighted_set[feature_id, j_sample].todense())[:, 0]
            fingerprint_k = np.zeros((1, self.dimension_num))[0]
            fingerprint_y = np.zeros((1, self.dimension_num))[0]

            fingerprints.GenerateFingerprintOfInstance(self.dimension_num,
                                                       feature_weight, feature_id, feature_id_num, self.seed * repeat,
                                                       fingerprint_k, fingerprint_y)

            fingerprints_k[j_sample, :] = fingerprint_k
            fingerprints_y[j_sample, :] = fingerprint_y

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def cws(self, repeat=1):
        """The Consistent Weighted Sampling (CWS) algorithm, as the first of the Consistent Weighted Sampling scheme,
           extends "active indices" from $[0, S]$ in [Gollapudi et. al., 2006](1) to $[0, +\infty]$.
           M. Manasse, F. McSherry, and K. Talwar, "Consistent Weighted Sampling", Unpublished technical report, 2010.

        Parameters
        -----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix

        Notes
        ----------
        The operations of seeking "active indices" and computing hashing values are implemented by C++
        due to low efficiency of Python. The operations cannot be vectorized in Python so that it would be
        very slow.
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        for j_sample in range(0, self.instance_num):

            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            feature_id_num = feature_id.shape[0]

            fingerprints = CDLL('./cpluspluslib/cws_fingerprints.so')
            fingerprints.GenerateFingerprintOfInstance.argtypes = [c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_int, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   c_int, c_int,
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS"),
                                                                   np.ctypeslib.ndpointer(dtype=c_double, ndim=1,
                                                                                          flags="C_CONTIGUOUS")]
            fingerprints.GenerateFingerprintOfInstance.restype = None
            weights = np.array(self.weighted_set[feature_id, j_sample].todense())[:, 0]
            fingerprint_k = np.zeros((1, self.dimension_num))[0]
            fingerprint_y = np.zeros((1, self.dimension_num))[0]

            fingerprints.GenerateFingerprintOfInstance(self.dimension_num,
                                                       weights, feature_id, feature_id_num, self.seed * repeat,
                                                       fingerprint_k, fingerprint_y)

            fingerprints_k[j_sample, :] = fingerprint_k
            fingerprints_y[j_sample, :] = fingerprint_y

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def icws(self, repeat=1):
        """The Improved Consistent Weighted Sampling (ICWS) algorithm, directly samples the two special "active indices",
           $y_k$ and $z_k$.
           S. Ioffe, "Improved Consistent Weighted Sampling, Weighted Minhash and L1 Sketching",
           in ICDM, 2010, pp. 246-255.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            t_matrix = np.floor(np.divide(
                np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
                gamma) + beta[feature_id, :])
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(np.multiply(-np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])),
                                             np.multiply(u1[feature_id, :], u2[feature_id, :])), y_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def licws(self, repeat=1):
        """The 0-bit Consistent Weighted Sampling (0-bit CWS) algorithm generates the original hash code $(k, y_k)$
           by running ICWS, but finally adopts only $k$ to constitute the fingerprint.
           P. Li, "0-bit Consistent Weighted Sampling", in KDD, 2015, pp. 665-674.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            t_matrix = np.floor(np.divide(
                np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
                gamma) + beta[feature_id, :])
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(np.multiply(-np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])),
                                             np.multiply(u1[feature_id, :], u2[feature_id, :])), y_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def pcws(self, repeat=1):
        """The Practical Consistent Weighted Sampling (PCWS) algorithm improves the efficiency of ICWS
           by simplifying the mathematical expressions.
           W. Wu, B. Li, L. Chen, and C. Zhang, "Consistent Weighted Sampling Made More Practical",
           in WWW, 2017, pp. 1035-1043.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            gamma = - np.log(np.multiply(u1[feature_id, :], u2[feature_id, :]))
            t_matrix = np.floor(np.divide(
                np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1, self.dimension_num),
                gamma) + beta[feature_id, :])
            y_matrix = np.exp(np.multiply(gamma, t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(-np.log(x[feature_id, :]), np.divide(y_matrix, u1[feature_id, :]))

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def ccws(self, repeat=1, scale=1):
        """The Canonical Consistent Weighted Sampling (CCWS) algorithm directly uniformly discretizes the original weight
           instead of uniformly discretizing the logarithm of the weight as ICWS.
           W. Wu, B. Li, L. Chen, and C. Zhang, "Canonical Consistent Weighted Sampling for Real-Value Weighetd Min-Hash",
           in ICDM, 2016, pp. 1287-1292.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        scale: int
            a constant to adapt the weight

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        gamma = np.random.beta(2, 1, (self.feature_num, self.dimension_num))
        c = np.random.gamma(2, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            t_matrix = np.floor(scale * np.divide(np.matlib.repmat(self.weighted_set[feature_id, j_sample].todense(), 1,
                                                                   self.dimension_num),
                                                  gamma[feature_id, :]) + beta[feature_id, :])
            y_matrix = np.multiply(gamma[feature_id, :], (t_matrix - beta[feature_id, :]))
            a_matrix = np.divide(c[feature_id, :], y_matrix) - 2 * np.multiply(gamma[feature_id, :], c[feature_id, :])

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]
            fingerprints_y[j_sample, :] = y_matrix[min_position, np.arange(a_matrix.shape[1])]

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def i2cws(self, repeat=1):
        """The Improved Improved Consistent Weighted Sampling (I$^2$CWS) algorithm, samples the two special
           "active indices", $y_k$ and $z_k$, independently by avoiding the equation of $y_k$ and $z_k$ in ICWS.
           W. Wu, B. Li, L. Chen, C. Zhang and P. S. Yu, "Improved Consistent Weighted Sampling Revisited",
           DOI: 10.1109/TKDE.2018.2876250, 2018.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        -----------
        fingerprints_k: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        fingerprints_y: ndarray, shape (n_instances, dimension_num)
            one component of hash codes $(k, y_k)$ for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints_k = np.zeros((self.instance_num, self.dimension_num))
        fingerprints_y = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        beta1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        beta2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u3 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        u4 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v1 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))
        v2 = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]

            r2 = - np.log(np.multiply(u3[feature_id, :], u4[feature_id, :]))
            t_matrix = np.floor(np.divide(np.matlib.repmat(np.log(self.weighted_set[feature_id, j_sample].todense()), 1,
                                                           self.dimension_num), r2) + beta2[feature_id, :])
            z_matrix = np.exp(np.multiply(r2, (t_matrix - beta2[feature_id, :] + 1)))
            a_matrix = np.divide(- np.log(np.multiply(v1[feature_id, :], v2[feature_id, :])), z_matrix)

            min_position = np.argmin(a_matrix, axis=0)
            fingerprints_k[j_sample, :] = feature_id[min_position]

            r1 = - np.log(np.multiply(u1[feature_id[min_position], :], u2[feature_id[min_position], :]))
            gamma1 = np.array([-np.log(np.diag(r1[0]))])

            b = np.array([np.diag(beta1[feature_id[min_position], :][0])])
            t_matrix = np.floor(np.divide(np.log(np.transpose(self.weighted_set[feature_id[min_position], j_sample]
                                                              .todense())), gamma1) + b)
            fingerprints_y[j_sample, :] = np.exp(np.multiply(gamma1, (t_matrix - b)))

        elapsed = time.time() - start

        return fingerprints_k, fingerprints_y, elapsed

    def chum(self, repeat=1):
        """[Chum et. al., 2008] samples an element proportionally to its weight via an exponential distribution
           parameterized with the weight.
           O. Chum, J. Philbin, A. Zisserman, "Near Duplicate Image Detection: Min-Hash and Tf-Idf Weighting",
           in BMVC, vol. 810, 2008, pp. 812-815

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))
        start = time.time()

        x = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            feature_id = sparse.find(self.weighted_set[:, j_sample] > 0)[0]
            k_hash = np.divide(-np.log(x[feature_id, :]),
                               np.matlib.repmat(self.weighted_set[feature_id, j_sample].todense(), 1, self.dimension_num))

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def gollapudi2(self, repeat=1):
        """[Gollapudi et. al., 2006](2) preserves each weighted element by thresholding normalized real-valued weights
           with random samples.
           S. Gollapudi and R. Panigraphy, "Exploiting Asymmetry in Hierarchical Topic Extraction",
           in CIKM, 2006, pp. 475-482.

        Parameters
        ----------
        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """

        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        np.random.seed(self.seed * np.power(2, repeat - 1))

        start = time.time()
        hash_parameters = np.random.randint(1, self.C_PRIME, (self.dimension_num, 2))
        u = np.random.uniform(0, 1, (self.feature_num, self.dimension_num))

        for j_sample in range(0, self.instance_num):
            max_f = np.max(self.weighted_set[:, j_sample])

            feature_id = sparse.find(u <= np.divide(self.weighted_set[:, j_sample], max_f))[0]
            feature_id_num = feature_id.shape[0]
            k_hash = np.mod(
                np.dot(np.transpose(np.array([feature_id])), np.array([np.transpose(hash_parameters[:, 1])])) +
                np.dot(np.ones((feature_id_num, 1)), np.array([np.transpose(hash_parameters[:, 1])])),
                self.C_PRIME)

            min_position = np.argmin(k_hash, axis=0)
            fingerprints[j_sample, :] = feature_id[min_position]

        elapsed = time.time() - start

        return fingerprints, elapsed

    def shrivastava(self, repeat=1, scale=1):
        """[Shrivastava, 2016] uniformly samples the area which is composed of the upper bound of each element
           in the universal set by simulating rejection sampling.
           A. Shrivastava, "Simple and Efficient Weighted Minwise Hashing", in NIPS, 2016, pp. 1498-1506.

        Parameters
        ----------
        scale: int
            a constant to adapt the weight

        repeat: int, default: 1
            the number of repeating the algorithm as the part of the seed of the random number generator

        Returns
        ----------
        fingerprints: ndarray, shape (n_instances, dimension_num)
            hash codes for data matrix, where row represents a data instance

        elapsed: float
            time of hashing data matrix
        """
        fingerprints = np.zeros((self.instance_num, self.dimension_num))

        start = time.time()

        bound = np.ceil(np.max(self.weighted_set * scale, 1).todense()).astype(int)
        m_max = np.sum(bound)
        seed = np.arange(1, self.dimension_num+1)

        comp_to_m = np.zeros((1, self.feature_num), dtype=int)
        int_to_comp = np.zeros((1, m_max), dtype=int)
        i_dimension = 0
        for i in range(0, m_max):
            if i == comp_to_m[0, i_dimension] and i_dimension < self.feature_num-1:
                i_dimension = i_dimension + 1
                comp_to_m[0, i_dimension] = comp_to_m[0, i_dimension - 1] + bound[i_dimension - 1, 0]
            int_to_comp[0, i] = i_dimension - 1

        for j_sample in range(0, self.instance_num):
            instance = (scale * self.weighted_set[:, j_sample]).todense()

            for d_id in range(0, self.dimension_num):
                np.random.seed(seed[d_id] * np.power(2, repeat - 1))
                while True:
                    rand_num = np.random.uniform(1, m_max)
                    rand_floor = np.floor(rand_num).astype(int)
                    comp = int_to_comp[0, rand_floor]
                    if rand_num <= comp_to_m[0, comp] + instance[comp]:
                        break
                    fingerprints[j_sample, d_id] = fingerprints[j_sample, d_id] + 1

        elapsed = time.time() - start

        return fingerprints, elapsed
