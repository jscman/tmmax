from typing import Union, Tuple
import jax.numpy as jnp

def _fresnel_s(_first_layer_n: Union[float, jnp.ndarray], 
               _second_layer_n: Union[float, jnp.ndarray],
               _first_layer_theta: Union[float, jnp.ndarray], 
               _second_layer_theta: Union[float, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


    """
    
    # Calculate the reflection coefficient for s-polarized light using Fresnel's equations.
    # The formula: r_s = (n1 * cos(theta1) - n2 * cos(theta2)) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is reflected at the interface.
    r_s = ((_first_layer_n * jnp.cos(_first_layer_theta) - _second_layer_n * jnp.cos(_second_layer_theta)) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Calculate the transmission coefficient for s-polarized light using Fresnel's equations.
    # The formula: t_s = 2 * n1 * cos(theta1) / (n1 * cos(theta1) + n2 * cos(theta2))
    # This measures how much of the light is transmitted through the interface.
    t_s = (2 * _first_layer_n * jnp.cos(_first_layer_theta) /
           (_first_layer_n * jnp.cos(_first_layer_theta) + _second_layer_n * jnp.cos(_second_layer_theta)))
    
    # Return the reflection and transmission coefficients as a JAX array
    return jnp.array([r_s, t_s])
