import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO

class LanderFinalDefinitivoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pasos = 0 
        self.contador_suelo = 0 # Para contar los 2 segundos
        
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
        
        contacto_izq = obs[6]
        contacto_der = obs[7]
        contacto_ambas = contacto_izq and contacto_der
        
        # Detectamos si está estable en el suelo
        # (Ambas patas tocan y velocidad vertical ridícula)
        esta_estable = contacto_ambas and abs(vel_y) < 0.05
        
        # --- LOGICA DEL TEMPORIZADOR DE SUELO (2 Segundos) ---
        if esta_estable:
            self.contador_suelo += 1
        else:
            self.contador_suelo = 0 # Si salta o rebota, reiniciamos la cuenta
            
        # Meta: 100 frames a 50fps = 2 segundos
        objetivo_cumplido = self.contador_suelo >= 100

        # -------------------------------------------------
        #                  RECOMPENSAS
        # -------------------------------------------------

        # A) SI AÚN ESTAMOS VOLANDO O ATERRIZANDO (No hemos cumplido los 2s)
        if not objetivo_cumplido:
            
            # 1. IMPUESTO DE TIEMPO (Solo aplica si NO está estable esperando)
            # Si ya está en la cuenta atrás de 2s, le perdonamos el tiempo
            # para que no se ponga nervioso.
            if self.contador_suelo == 0:
                factor_urgencia = (self.pasos / 100.0)
                reward -= 0.1 * factor_urgencia
            else:
                # Si está esperando, le damos un caramelito para que aguante quieto
                reward += 0.5 

            # 2. EN EL AIRE (Lógica agresiva anterior)
            if not contacto_izq and not contacto_der:
                # Embudo
                reward -= abs(pos_x) * 4.0
                
                # Anti-flotación (Si estás centrado, BAJA)
                if abs(pos_x) < 0.2 and vel_y > -0.2:
                    reward -= 2.0 * (self.pasos / 100.0)
                
                # Incentivo bajada
                if -1.5 < vel_y < -0.5:
                    reward += 0.5

            # 3. EN EL SUELO (Pero aún no han pasado los 2s)
            if esta_estable:
                 # Castigo si intenta mover la nave mientras espera
                 if action != 0:
                     reward -= 0.5
        
        # B) FINALIZACIÓN (Pasaron los 2 segundos)
        else:
            # ¡SE ACABÓ!
            terminated = True
            
            # Calculamos el premio final
            if abs(pos_x) < 0.1:
                reward += 150.0 # Diana perfecta
            elif abs(pos_x) < 0.2:
                reward += 75.0 # Aterrizaje válido
            else:
                reward -= 40.0 # Aterrizaje fuera de zona (raro si está estable)

        # Si el entorno original dice que acabó (por crash), respetamos eso
        if terminated and not objetivo_cumplido:
            # Si se estrelló fuerte antes de los 2s
            if not esta_estable:
                reward -= 20.0

        return obs, reward, terminated, truncated, info

def train_final_lander():
    env = gym.make("LunarLander-v3")
    env = LanderFinalDefinitivoWrapper(env)
    
    print("--- INICIANDO ENTRENAMIENTO FINAL (Con espera de 2s) ---")
    
    # Mismo learning rate agresivo
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0007)
    
    # 150k pasos suelen bastar
    model.learn(total_timesteps=300000)
    
    print("--- Entrenamiento finalizado ---")
    return model

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
