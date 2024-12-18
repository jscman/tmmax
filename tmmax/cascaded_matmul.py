import jax
#jax.config.update('jax_enable_x64', True) # Ensure high precision (64-bit) is enabled in JAX
import jax.numpy as jnp # Import JAX's version of NumPy for differentiable computations
from jax import Array
from jax.typing import ArrayLike

def matmul(carry, phase_r_t):


    transfer_matrix_00 = jnp.exp(jnp.multiply(jnp.array([-1j], dtype = jnp.complex64), phase_r_t.at[0].get()))
    transfer_matrix_11 = jnp.exp(jnp.multiply(jnp.array([1j], dtype = jnp.complex64), phase_r_t.at[0].get()))
    transfer_matrix_01 = jnp.multiply(phase_r_t.at[1].get(), transfer_matrix_00)
    transfer_matrix_10 = jnp.multiply(phase_r_t.at[1].get(), transfer_matrix_11)

    transfer_matrix = jnp.multiply(jnp.true_divide(1, phase_r_t.at[2].get()), jnp.array([[transfer_matrix_00, transfer_matrix_01],
                                                                                         [transfer_matrix_10, transfer_matrix_11]]))

    result = jnp.matmul(carry, transfer_matrix)

    

    return jnp.squeeze(result), jnp.squeeze(result)  # Return the updated matrix and None as a placeholder for jax.lax.scan


#@jax.jit
def cascaded_matrix_multiplication(phases: ArrayLike, rts: ArrayLike) -> Array:
    """
    Calculates the angles of incidence across layers for a set of refractive indices (nk_list_2d)
    and an initial angle of incidence (initial_theta) using vectorization.

    Returns:
        jnp.ndarray: A 3D JAX array where the [i, j, :] entry represents the angles of incidence
                     for the j-th initial angle at the i-th wavelength. The size of the third dimension
                     corresponds to the number of layers.
    """
    phase_rt_stack = jnp.concat([jnp.expand_dims(phases, 1), rts], axis=1)
    """
    Performs cascaded matrix multiplication on a sequence of complex matrices using scan.

    Args:
        phases_ts_rs (jax.numpy.ndarray): An array of shape [N, 2, 2], where N is the number of 2x2 complex matrices.
                                          Each 2x2 matrix is represented by its 2x2 elements arranged in a 3D array.

    Returns:
        jax.numpy.ndarray: The final result of multiplying all the matrices together in sequence.
                           This result is a single 2x2 complex matrix representing the accumulated product of all input matrices.
    """
    initial_value = jnp.eye(2, dtype=jnp.complex64)
    # Initialize with the identity matrix of size 2x2. # The identity matrix acts as the multiplicative identity,
    # ensuring that the multiplication starts correctly.

    # jax.lax.scan applies a function across the sequence of matrices.
    #Here, _matmul is the function applied, starting with the identity matrix.
    # `result` will hold the final matrix after processing all input matrices.
    result, _ = jax.lax.scan(matmul, initial_value, phase_rt_stack)  # Scan function accumulates results of _matmul over the matrices.

    return result