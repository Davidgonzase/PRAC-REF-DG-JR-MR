import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO

class LanderChebyshevWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pasos = 0
        self.contador_suelo = 0
        # Definimos los pesos
        self.w_distancia = 3.0  
        self.w_velocidad = 5.0  
        self.w_angulo    = 2.0 

    def reset(self, **kwargs):
        self.pasos = 0
        self.contador_suelo = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.pasos += 1

        pos_x = obs[0]
        pos_y = obs[1]
        vel_x = obs[2]
        vel_y = obs[3]
        angulo = obs[4]
        vel_angular = obs[5]
        contacto_izq = obs[6]
        contacto_der = obs[7]

        contacto_ambas = contacto_izq and contacto_der
        # Solo contamos el aterrizaje si toca con ambas patas y está casi quieto
        esta_estable = contacto_ambas and abs(vel_y) < 0.05

        if esta_estable:
            self.contador_suelo += 1
        else:
            self.contador_suelo = 0

        objetivo_cumplido = self.contador_suelo >= 100

        error_distancia = abs(pos_x)  
        error_angulo    = abs(angulo) 
        error_velocidad = np.sqrt(vel_x**2 + vel_y**2)

        p_dist = error_distancia * self.w_distancia
        p_vel  = error_velocidad * self.w_velocidad
        p_ang  = error_angulo    * self.w_angulo

        # Penalizamos basándonos en el PEOR error actual.
        # Obliga al agente a arreglar su mayor defecto en cada paso.
        worst_error = max(p_dist, p_vel, p_ang)
        step_reward = -worst_error + 0.1

        if not objetivo_cumplido:
            reward = step_reward

            if terminated:
                reward = -10.0 # Castigo fuerte si se estrella

        else:
            terminated = True 

            # Premios finales según precisión: cuanto más centrado, más puntos.
            if abs(pos_x) < 0.1:
                reward = 100.0 
            elif abs(pos_x) < 0.2:
                reward = 50.0  
            else:
                reward = 10.0  

        return obs, reward, terminated, truncated, info

def train_final_lander():
    env = gym.make("LunarLander-v3")
    env = LanderChebyshevWrapper(env)

    print("--- INICIANDO ENTRENAMIENTO (Chebyshev Scalarization) ---")

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0007)
    model.learn(total_timesteps=300000)

    print("--- Entrenamiento finalizado ---")
    return model

def save_video(model, filename="lunar_lander_chebyshev.mp4"):
    print(f"--- Grabando video: {filename} ---")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = LanderChebyshevWrapper(env)

    obs, _ = env.reset()
    frames = []

    done = False
    truncated = False
    while not (done or truncated):
        frames.append(env.render())
        # Usamos deterministic=True para ver lo mejor que aprendió 
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)

    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print("¡Video listo!")

if __name__ == "__main__":
    agent = train_final_lander()
    save_video(agent)
