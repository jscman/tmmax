import jax.numpy as jnp
from jax import vmap
import jax
import numpy as np
import pickle
import sys
from jax import Array
from jax.typing import ArrayLike

# Define EPSILON as the smallest representable positive number such that 1.0 + EPSILON != 1.0
EPSILON = sys.float_info.epsilon

def is_forward_if_bigger_than_eps_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))
    # For evanescent or lossy mediums, forward is determined by decay
    is_forward_s = jnp.invert(jnp.signbit(jnp.imag(n_cos_theta)))
    return is_forward_s

def is_forward_if_smaller_than_eps_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))
    # For s-polarization: Re[n cos(theta)] > 0
    is_forward_s = jnp.invert(jnp.signbit(jnp.real(n_cos_theta)))
    return is_forward_s

def is_forward_if_bigger_than_eps_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))
    # For evanescent or lossy mediums, forward is determined by decay
    is_forward_p = jnp.invert(jnp.signbit(jnp.imag(n_cos_theta)))
    # The decay condition applies to both polarizations (s and p, so we return s and p  as s and s) equally
    return is_forward_p

def is_forward_if_smaller_than_eps_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # For p-polarization: Re[n cos(theta*)] > 0
    n_cos_theta_star = jnp.multiply(n, jnp.cos(jnp.conj(theta)))
    is_forward_p = jnp.invert(jnp.signbit(jnp.real(n_cos_theta_star)))
    return is_forward_p

def is_forward_if_bigger_than_eps_u_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))
    # For evanescent or lossy mediums, forward is determined by decay
    is_forward_s = jnp.invert(jnp.signbit(jnp.imag(n_cos_theta)))
    # The decay condition applies to both polarizations (s and p, so we return s and p  as s and s) equally
    return is_forward_s, is_forward_s

def is_forward_if_smaller_than_eps_u_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Calculate n * cos(theta) to evaluate propagation direction for s-polarization
    n_cos_theta = jnp.multiply(n, jnp.cos(theta))
    # For s-polarization: Re[n cos(theta)] > 0
    is_forward_s = jnp.invert(jnp.signbit(jnp.real(n_cos_theta)))
    # For p-polarization: Re[n cos(theta*)] > 0
    n_cos_theta_star = jnp.multiply(n, jnp.cos(jnp.conj(theta)))
    is_forward_p = jnp.invert(jnp.signbit(jnp.real(n_cos_theta_star)))
    return is_forward_s, is_forward_p

def is_propagating_wave_s_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Handle the evanescent and lossy cases by checking the imaginary part
    condition = jnp.squeeze(jnp.greater(jnp.abs(jnp.imag(jnp.multiply(n, jnp.cos(theta)))), jnp.multiply(jnp.array([EPSILON]), jnp.array([1e3]))))
    # Return based on polarization argument
    is_forward_s = jax.lax.cond(condition, is_forward_if_bigger_than_eps_s_pol, is_forward_if_smaller_than_eps_s_pol, n, theta)
    # s-polarization case
    return is_forward_s

def is_propagating_wave_p_pol(n: ArrayLike, theta: ArrayLike) -> Array:
    # Handle the evanescent and lossy cases by checking the imaginary part
    condition = jnp.squeeze(jnp.greater(jnp.abs(jnp.imag(jnp.multiply(n, jnp.cos(theta)))), jnp.multiply(jnp.array([EPSILON]), jnp.array([1e3]))))
    # Return based on polarization argument
    is_forward_p = jax.lax.cond(condition, is_forward_if_bigger_than_eps_p_pol, is_forward_if_smaller_than_eps_p_pol, n, theta)
    # p-polarization case
    return is_forward_p

def update_theta_arr_incoming(theta_array):
    return theta_array.at[0].set(jnp.pi - theta_array.at[0].get())

def update_theta_arr_outgoing(theta_array):
    return theta_array.at[-1].set(jnp.pi - theta_array.at[-1].get())

def return_unchanged_theta(theta_array):
    return theta_array

def compute_layer_angles_s_pol(angle_of_incidence: ArrayLike,
                               nk_list: ArrayLike) -> Array:
    #print("angle_of_incidence : ", jnp.asarray(angle_of_incidence))
    #print("nk_list : ", jnp.asarray(nk_list))
    # Calculate the sine of the angles in the first layer using Snell's law
    sin_theta = jnp.true_divide(jnp.multiply(jnp.sin(angle_of_incidence), nk_list.at[0].get()), nk_list)
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is preferred for compatibility with complex values if needed
    theta_array = jnp.arcsin(sin_theta)
    #print("first theta : ", jnp.asarray(theta_array))
    # If the angle is not forward-facing, we subtract it from pi to flip the orientation.
    incoming_props = is_propagating_wave_s_pol(nk_list.at[0].get(), theta_array.at[0].get())
    outgoing_props = is_propagating_wave_s_pol(nk_list.at[-1].get(), theta_array.at[-1].get())

    # Handle the evanescent and lossy cases by checking the imaginary part
    condition_incoming = jnp.array_equal(False, jnp.array([True], dtype=bool))
    condition_outgoing = jnp.array_equal(False, jnp.array([True], dtype=bool))

    theta_array = jax.lax.cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=theta_array)
    #print("second theta : ", jnp.asarray(theta_array))
    theta_array = jax.lax.cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=theta_array)
    #print("third theta : ", jnp.asarray(theta_array))

    # Return a 1D theta array for each layer
    return theta_array

def compute_layer_angles_p_pol(angle_of_incidence: ArrayLike,
                               nk_list: ArrayLike) -> Array:

    # Calculate the sine of the angles in the first layer using Snell's law
    sin_theta = jnp.true_divide(jnp.multiply(jnp.sin(angle_of_incidence), nk_list.at[0].get()), nk_list)
    # Compute the angle (theta) in each layer using the arcsin function
    # jnp.arcsin is preferred for compatibility with complex values if needed
    theta_array = jnp.arcsin(sin_theta)
    # If the angle is not forward-facing, we subtract it from pi to flip the orientation.
    incoming_props = is_propagating_wave_p_pol(nk_list.at[0].get(), theta_array.at[0].get())
    outgoing_props = is_propagating_wave_p_pol(nk_list.at[-1].get(), theta_array.at[-1].get())

    # Handle the evanescent and lossy cases by checking the imaginary part
    condition_incoming = jnp.array_equal(incoming_props, jnp.array([True], dtype=bool))
    condition_outgoing = jnp.array_equal(outgoing_props, jnp.array([True], dtype=bool))

    theta_array = jax.lax.cond(condition_incoming, update_theta_arr_incoming, return_unchanged_theta, operand=theta_array)
    theta_array = jax.lax.cond(condition_outgoing, update_theta_arr_outgoing, return_unchanged_theta, operand=theta_array)

    # Return a 1D theta array for each layer
    return theta_array

#@jit
def compute_layer_angles(angle_of_incidence: ArrayLike,
                         nk_list: ArrayLike,
                         polarization: ArrayLike) -> Array:

    return jnp.select(condlist=[jnp.array_equal(polarization, jnp.array([0], dtype = jnp.int16)),
                                jnp.array_equal(polarization, jnp.array([1], dtype = jnp.int16))],
                    choicelist=[compute_layer_angles_s_pol(angle_of_incidence, nk_list),
                                compute_layer_angles_p_pol(angle_of_incidence, nk_list)])