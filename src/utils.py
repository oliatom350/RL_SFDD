import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


##############################################
##########    PLOT FUNCTIONS    ##############
##############################################

def plot(list_stats):
  # Creamos una lista de índices para el eje x
  indices = list(range(len(list_stats)))

  # Creamos el gráfico
  plt.figure(figsize=(6, 3))
  plt.plot(indices, list_stats)

  # Añadimos título y etiquetas
  plt.title('Proporción de recompensas')
  plt.xlabel('Episodio')
  plt.ylabel('Proporción')

  # Mostramos el gráfico
  plt.grid(True)
  plt.show()


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def plot_episode_rewards(envs, rolling_length = 500):
    plt.figure(figsize=(30, 10))
    for name, env in envs:
        reward_moving_average = get_moving_avgs(
            env.return_queue,
            rolling_length,
            "valid"
        )
        plt.plot(range(len(reward_moving_average)), reward_moving_average, label=name)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Evolución de la recompensa de los Episodios")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_episode_lengths(envs, rolling_length = 500):
    plt.figure(figsize=(30, 10))
    for name, env in envs:
        length_moving_average = get_moving_avgs(
            env.length_queue,
            rolling_length,
            "valid"
        )
        plt.plot(range(len(length_moving_average)), length_moving_average, label=name)
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Evolución de la recompensa de los Episodios")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_episode_lengths(episode_lengths):
    """Grafica la longitud de los episodios a lo largo del tiempo."""
    plt.figure(figsize=(30, 10))
    plt.plot(range(len(episode_lengths)), episode_lengths, label="Longitud del episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Longitud")
    plt.title("Evolución de la Longitud de los Episodios")
    plt.grid(True)
    plt.legend()
    plt.show()


##############################################
###########    GYM FUNCTIONS    ##############
##############################################

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