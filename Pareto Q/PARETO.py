import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO

# Wrapper Unificado: Combina precisión de centrado con estabilidad de 2 segundos
class ParetoLanderUnificado(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pasos = 0 
        self.contador_suelo = 0
        
    def reset(self, **kwargs):
        self.pasos = 0
        self.contador_suelo = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.pasos += 1
        
        # Variables de observación
        pos_x = obs[0]
        vel_y = obs[3]
        angulo = obs[4]
        contacto_izq = obs[6]
        contacto_der = obs[7]
        
        # Definición de estabilidad (Ambas patas y velocidad vertical casi nula)
        esta_estable = (contacto_izq and contacto_der) and abs(vel_y) < 0.05
        
        # Gestión del temporizador de 2 segundos (100 frames a 50fps)
        if esta_estable:
            self.contador_suelo += 1
        else:
            self.contador_suelo = 0
            
        objetivo_cumplido = self.contador_suelo >= 100


        if not objetivo_cumplido:
            reward -= abs(pos_x) * 4.0      # Fuerza el centrado en X
            reward -= abs(angulo) * 0.5     # Fuerza la verticalidad
            
            # 2. Gestión de vuelo y urgencia
            if self.contador_suelo == 0:
                reward -= 0.1 * (self.pasos / 100.0) # Penalización por tiempo
                
                # Incentivo de descenso controlado
                if -1.5 < vel_y < -0.5:
                    reward += 0.5
            else:
                reward += 0.5  # Premio por mantener la estabilidad en el suelo


            if esta_estable and action != 0:
                reward -= 0.5
        
        else:
            terminated = True
            
            # Bonus final por precisión
            distancia = abs(pos_x)
            if distancia < 0.05:
                reward += 150.0
            elif distancia < 0.15:
                reward += 75.0
            else:
                reward -= 40.0

        # Penalización si el episodio termina por crash (antes de los 2s)
        if terminated and not objetivo_cumplido:
            if not esta_estable:
                reward -= 20.0

        return obs, reward, terminated, truncated, info

def train_pareto_unificado():
    env = gym.make("LunarLander-v3")
    env = ParetoLanderUnificado(env)
    
    print("--- INICIANDO ENTRENAMIENTO FINAL ---")
    
    # Se utiliza un Learning Rate intermedio para estabilidad y rapidez
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005)
    model.learn(total_timesteps=300000)
    
    print("--- Entrenamiento finalizado ---")
    return model

def save_video(model, filename="pareto_unificado_lander.mp4"):
    print(f"--- Grabando video: {filename} ---")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = ParetoLanderUnificado(env)
    
    obs, _ = env.reset()
    frames = []
    
    done = False
    truncated = False
    while not (done or truncated):
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
    
    env.close()
    imageio.mimsave(filename, frames, fps=30)
    print(f"¡Video {filename} guardado!")

if __name__ == "__main__":
    agent = train_pareto_unificado()
    save_video(agent)