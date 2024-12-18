import jax.numpy as jnp
from jax import Array
from jax import jit
from jax.typing import ArrayLike

def fresnel_s(first_layer_n: ArrayLike,
              second_layer_n: ArrayLike,
              first_layer_theta: ArrayLike,
              second_layer_theta: ArrayLike) -> Array:
    """
    This function calculates the Fresnel reflection (r_s) and transmission (t_s) coefficients
    for s-polarized light (electric field perpendicular to the plane of incidence) at the interface
    between two materials. The inputs are the refractive indices and the angles of incidence and
    refraction for the two layers.

    Args:
        _first_layer_n (Union[float, jnp.ndarray]): Refractive index of the first layer (incident medium).
            Can be a float or an array if computing for multiple incident angles/materials.
        _second_layer_n (Union[float, jnp.ndarray]): Refractive index of the second layer (transmitted medium).
            Similar to the first argument, this can also be a float or an array.
        _first_layer_theta (Union[float, jnp.ndarray]): Angle of incidence in the first layer (in radians).
            Can be a float or an array.
        _second_layer_theta (Union[float, jnp.ndarray]): Angle of refraction in the second layer (in radians).
            Can be a float or an array.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two jax.numpy arrays:
            - r_s: The Fresnel reflection coefficient for s-polarized light.
            - t_s: The Fresnel transmission coefficient for s-polarized light.
    """
    cos_first_theta = jnp.cos(first_layer_theta)
    cos_second_theta = jnp.cos(second_layer_theta)
    first_ncostheta = jnp.multiply(first_layer_n, cos_first_theta)
    second_ncostheta = jnp.multiply(second_layer_n, cos_second_theta)
    add_ncosthetas = jnp.add(first_ncostheta, second_ncostheta)
    # Calculate the reflection coefficient for s-polarized light using Fresnel's equations.
    # The formula: r_s = (n1 * cos(theta1) - n2 * cos(theta2)) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is reflected at the interface.

    r_s = jnp.true_divide(jnp.subtract(first_ncostheta, second_ncostheta), add_ncosthetas)

    # Calculate the transmission coefficient for s-polarized light using Fresnel's equations.
    # The formula: t_s = 2 * n1 * cos(theta1) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is transmitted through the interface.
    t_s = jnp.true_divide(jnp.multiply(2,first_ncostheta),add_ncosthetas)

    # Return the reflection and transmission coefficients as a JAX array
    return jnp.stack([r_s, t_s])


def fresnel_p(first_layer_n: ArrayLike,
              second_layer_n: ArrayLike,
              first_layer_theta: ArrayLike,
              second_layer_theta: ArrayLike) -> Array:
    """
    This function calculates the Fresnel reflection (r_p) and transmission (t_p) coefficients
    for p-polarized light at the interface between two different media. It uses the refractive indices
    of the two media (_first_layer_n and _second_layer_n) and the incident and transmitted angles
    (_first_layer_theta and _second_layer_theta) to compute these values.

    Args:
        _first_layer_n: Refractive index of the first medium (can be float or ndarray).
        _second_layer_n: Refractive index of the second medium (can be float or ndarray).
        _first_layer_theta: Incident angle (in radians) in the first medium (can be float or ndarray).
        _second_layer_theta: Transmitted angle (in radians) in the second medium (can be float or ndarray).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing two arrays:
            - r_p: The reflection coefficient for p-polarized light.
            - t_p: The transmission coefficient for p-polarized light.
    """
    cos_first_theta = jnp.cos(first_layer_theta)
    cos_second_theta = jnp.cos(second_layer_theta)
    second_n_first_costheta = jnp.multiply(second_layer_n, cos_first_theta)
    first_n_second_costheta = jnp.multiply(first_layer_n, cos_second_theta)
    add_ncosthetas = jnp.add(second_n_first_costheta, first_n_second_costheta)
    # Calculate the reflection coefficient for p-polarized light (r_p)
    # This equation is based on the Fresnel equations for p-polarization, where
    # r_p is the ratio of the reflected and incident electric field amplitudes for p-polarized light.
    r_p = jnp.true_divide(jnp.subtract(second_n_first_costheta, first_n_second_costheta), add_ncosthetas)


    # Calculate the transmission coefficient for p-polarized light (t_p)
    # This equation is also derived from the Fresnel equations for p-polarization.
    # t_p represents the ratio of the transmitted and incident electric field amplitudes.
    t_p = jnp.true_divide(jnp.multiply(2,jnp.multiply(first_layer_n, cos_first_theta)),add_ncosthetas)

    # Return the reflection and transmission coefficients as a tuple of jnp arrays
    # Both r_p and t_p are essential for understanding how light interacts with different layers.
    return jnp.stack([r_p, t_p])