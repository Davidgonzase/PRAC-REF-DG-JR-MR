import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO

# Wrapper final para LunarLander con enfoque en aterrizaje rápido y preciso
class LanderFinalDefinitivoWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.pasos = 0 
        self.contador_suelo = 0
       
    def reset(self, **kwargs):
        self.pasos = 0
        self.contador_suelo = 0
        return self.env.reset(**kwargs)
    
    # Paso modificado con los enfoques finales siendo estos rápido aterrizaje, precisiónm, estabilidad lateral y bonificacion por inactividad al aterrizar
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.pasos += 1
        
        pos_x = obs[0]
        pos_y = obs[1]
        vel_x = obs[2]
        vel_y = obs[3]
        angulo = obs[4]
        
        # Se define un aterrizaje estable como estar en contacto con ambas patas y velocidad vertical baja
        contacto_izq = obs[6]
        contacto_der = obs[7]
        contacto_ambas = contacto_izq and contacto_der
        
        esta_estable = contacto_ambas and abs(vel_y) < 0.05
        
        # Objetivo cumplido tras 2 segundos estable (100 frames)
        if esta_estable:
            self.contador_suelo += 1
        else:
            self.contador_suelo = 0
            
        objetivo_cumplido = self.contador_suelo >= 100

        if not objetivo_cumplido:
            if self.contador_suelo == 0:
                factor_urgencia = (self.pasos / 100.0)
                reward -= 0.1 * factor_urgencia
            else:
                reward += 0.5 

            # Penalizaciones y recompensas por control lateral y descenso
            if not contacto_izq and not contacto_der:
                reward -= abs(pos_x) * 4.0
                
                if abs(pos_x) < 0.2 and vel_y > -0.2:
                    reward -= 2.0 * (self.pasos / 100.0)
                
                if -1.5 < vel_y < -0.5:
                    reward += 0.5

            if esta_estable:
                 if action != 0:
                     reward -= 0.5
        
        else:
            terminated = True
            
            # Bonificación por aterrizaje exitoso y preciso
            
            if abs(pos_x) < 0.1:
                reward += 150.0
            elif abs(pos_x) < 0.2:
                reward += 75.0
            else:
                reward -= 40.0

        # Penalización por crash antes de cumplir objetivo
        
        if terminated and not objetivo_cumplido:
            if not esta_estable:
                reward -= 20.0

        return obs, reward, terminated, truncated, info

# Entrenamiento del modelo final con el wrapper definitivo
def train_final_lander():
    env = gym.make("LunarLander-v3")
    env = LanderFinalDefinitivoWrapper(env)
    
    print("--- INICIANDO ENTRENAMIENTO FINAL ---")
    
    # En este caso se utiliza PPO con una tasa de aprendizaje ajustada
    # Como prueba se han realizado 300,000 timesteps, esto genera un modelo competente pero es capaz de buscar trucos para obtener mas recompensa aunque empeore otros aspectos
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0007)
    model.learn(total_timesteps=300000)
    
    print("--- Entrenamiento finalizado ---")
    return model

# Grabación de video del modelo entrenado
def save_video(model, filename="lunar_lander_final_2s.mp4"):
    print(f"--- Grabando video: {filename} ---")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = LanderFinalDefinitivoWrapper(env)
    
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
    print("¡Video listo!")

if __name__ == "__main__":
    agent = train_final_lander()
    save_video(agent)