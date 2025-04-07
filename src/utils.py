import numpy as np
import gymnasium as gym

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def make_env(env_name, **kwargs):
    """
    Crea un entorno Gymnasium basado en el nombre proporcionado y los parámetros específicos.

    :param env_name: Nombre del entorno (en nuestro proyecto, "FrozenLake-v1" o "CartPole-v1")
    :param kwargs: Parámetros adicionales específicos para cada entorno
    :return: Instancia del entorno de Gymnasium
    """
    # Diccionario con entornos y sus parámetros predeterminados
    env_params = {
        "FrozenLake-v1": {
            "map_name": kwargs.get("map_name", "4x4"),
            "is_slippery": kwargs.get("is_slippery", False),
            "render_mode": kwargs.get("render_mode", 'ansi')
        },
        "CartPole-v1": {
            "render_mode": kwargs.get("render_mode", "rgb_array"),
            "max_episode_steps": kwargs.get("max_episode_steps", 500)
        }
    }

    # Comprobamos si el entorno es válido
    if env_name not in env_params:
        raise ValueError(f"El entorno '{env_name}' no es válido.")
    
    # Crear y devolver el entorno con los parámetros apropiados
    env = gym.make(env_name, **env_params[env_name])

    env.reset(seed=kwargs.get("seed", 42))
    return env