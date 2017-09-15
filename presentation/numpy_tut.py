# slice comma notation
# -1 = infer
# ... = any number of :

# axis argument

# np.sum
# np.mean
# np.dot - designated operator @, multiplies together
# dot(a, b)[i, j, k, m] = sum(a[i, j, :] * b[k, :, m])
# same as matrix product for 2D arrays, inner product for 1D
# np.outer - np.dot(a[..., None], b[None, ...])
# np.inner - product along last dimension

# specifying datatype
# dtype=np.int32, np.int8, np.float32, np.float64

# broadcasting - 5 * np.arange().reshape()
# np.ones([1, 3]) * np.ones([2, 1])
