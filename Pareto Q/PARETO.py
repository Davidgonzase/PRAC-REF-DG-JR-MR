import gymnasium as gym
import numpy as np
import imageio
from stable_baselines3 import PPO

# --- CONFIGURACIÓN V19: EL PURISTA (CENTRO PURO, SIN RUIDO) ---

class ParetoPuristaWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_count = 0
        
    def step(self, action):
        # 1. DEJAMOS QUE EL JUEGO HAGA SU TRABAJO
        # El juego ya premia aterrizar y castiga gastar gasolina (velocidad implícita)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        pos_x = obs[0]
        angulo = obs[4]
        leg1, leg2 = obs[6], obs[7]
        
        # 2. INTERVENCIÓN MINIMALISTA (LO QUE TÚ PEDISTE)
        
        # A) EL IMÁN (Centrado Puro)
        # Castigamos fuertemente cualquier desviación del 0.0 en X.
        # Esto fuerza a la IA a corregir la trayectoria INMEDIATAMENTE.
        reward -= abs(pos_x) * 4.0 
        
        # B) EL NIVEL DE BURBUJA
        # Es imposible aterrizar en el centro puro si vas inclinado.
        reward -= abs(angulo) * 0.5

        # 3. EL GRAN BONUS FINAL (Solo al terminar)
        if terminated and not truncated:
            # Verificamos si ha aterrizado de verdad (ambas patas tocando)
            # El juego suele dar +100 o -100, usamos eso como guía, 
            # pero añadimos NUESTRO incentivo de precisión.
            if leg1 and leg2:
                distancia_centro = abs(pos_x)
                
                if distancia_centro < 0.05: # ¡DIANA PERFECTA!
                    reward += 100.0
                    print(f"🎯 BULLSEYE! Distancia: {distancia_centro:.3f}")
                elif distancia_centro < 0.15: # MUY BIEN
                    reward += 50.0
                else:
                    # Si aterriza pero lejos, le quitamos un poco de alegría
                    reward -= 20.0 

        # LOGS SIMPLES
        if terminated or truncated:
            self.episode_count += 1
            if self.episode_count % 20 == 0:
                estado = "WIN  " if (leg1 and leg2) else "CRASH"
                print(f"{self.episode_count:<5} | {estado} | X_Final:{pos_x:6.3f} | REWARD:{reward:6.1f}")

        return obs, reward, terminated, truncated, info

def train_pareto_purista():
    env = gym.make("LunarLander-v3")
    env = ParetoPuristaWrapper(env)
    
    print(f"--- ENTRENANDO: PARETO V19 (EL PURISTA) ---")
    # Learning Rate estándar, sin trucos. Dejamos que PPO haga su magia.
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003)
    
    # 300k pasos para que afine la puntería milimétrica
    model.learn(total_timesteps=300000)
    
    print("--- Fin del entrenamiento ---")
    return model

def save_video(model, filename="PARETO_lunar_lander_v19_purista.mp4"):
    print(f"--- Grabando video: {filename} ---")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = ParetoPuristaWrapper(env)
    
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
    agent = train_pareto_purista()
    save_video(agent)
