import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

# Wrapper multi-objetivo para LunarLander con 5 dimensiones de recompensa
class Lander5ObjWrapper(gym.Wrapper):
    def __init__(self, env, current_weights):
        super().__init__(env)
        self.pasos = 0 
        self.contador_suelo = 0
        self.current_weights = np.array(current_weights, dtype=np.float32)
        
    def update_weights(self, new_weights):
        self.current_weights = np.array(new_weights, dtype=np.float32)

    def reset(self, **kwargs):
        self.pasos = 0
        self.contador_suelo = 0
        return self.env.reset(**kwargs)
        
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.pasos += 1
        
        pos_x = obs[0]
        vel_y = obs[3]
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

        # Vector de recompensas: [Centrado, Tiempo, Descenso, Estabilidad, Aterrizaje]
        r_vec = np.zeros(5)

        if not objetivo_cumplido:
            
            # OBJ 0: Control lateral (Centrado)
            if not contacto_izq and not contacto_der:
                r_vec[0] = -abs(pos_x) * 0.05 

            # OBJ 2: Control de descenso
            if abs(pos_x) < 0.2 and vel_y > -0.2:
                r_vec[2] = -0.05  # Castigo por flotar
                
            if -1.5 < vel_y < -0.5:
                r_vec[2] += 0.01  # Premio por descenso controlado

            # OBJ 1: Eficiencia temporal
            if self.contador_suelo == 0:
                r_vec[1] = -0.002
            
            # OBJ 3: Estabilidad
            if esta_estable:
                r_vec[3] = 0.02
                if action != 0:
                     r_vec[3] -= 0.05  # Disciplina: no tocar controles cuando está estable
        
        else:
            # OBJ 4: Aterrizaje exitoso
            terminated = True
            if abs(pos_x) < 0.1:
                r_vec[4] = 1.0   # Aterrizaje centrado
            elif abs(pos_x) < 0.2:
                r_vec[4] = 0.5   # Aterrizaje aceptable
            else:
                r_vec[4] = -0.5  # Aterrizaje descentrado

        # Penalización por crash antes de cumplir objetivo
        if terminated and not objetivo_cumplido:
            if not esta_estable:
                r_vec[4] -= 1.0

        # Escalarización mediante producto punto
        scalar_reward = np.dot(r_vec, self.current_weights)
        info["vector_reward"] = r_vec
        
        return obs, scalar_reward, terminated, truncated, info

# Optimistic Linear Support: algoritmo para encontrar frontera de Pareto
class OLS:
    def __init__(self, num_objectives=5):
        self.num_obj = num_objectives
        self.ccs = []  # Convex Coverage Set
        self.visited_weights = []

    def add_solution(self, weights, value_vector, model_path):
        self.ccs.append({
            "weights": weights,
            "value": value_vector,
            "path": model_path
        })
        self.visited_weights.append(weights)
        print(f"[OLS] Solución agregada para w={np.round(weights, 2)}. Valor={np.round(value_vector, 2)}")

    def get_next_weight(self):
        # FASE 1: Explorar objetivos puros (esquinas del simplex)
        if len(self.ccs) < self.num_obj:
            next_w = np.zeros(self.num_obj)
            next_w[len(self.ccs)] = 1.0
            return next_w

        # FASE 2: Búsqueda OLS mediante maximin improvement
        print("[OLS] Calculando siguiente peso óptimo (Maximin Improvement)...")
        
        best_w = None
        max_improvement = -np.inf
        
        # Punto utópico: valor teórico máximo alcanzable en cada objetivo
        utopian_point = np.ones(self.num_obj) * 1.5 
        
        # Muestreo Monte Carlo en el espacio de pesos
        for _ in range(5000): 
            w = np.random.rand(self.num_obj)
            w /= np.sum(w)
            
            # Mejor valor actual con políticas existentes
            current_max_val = -np.inf
            for sol in self.ccs:
                val = np.dot(sol["value"], w)
                if val > current_max_val:
                    current_max_val = val
            
            # Valor optimista alcanzable
            optimistic_val = np.dot(utopian_point, w)
            
            # Potencial de mejora (regret)
            improvement_potential = optimistic_val - current_max_val
            
            # Filtro de diversidad: evitar pesos ya explorados
            dist_min = min([np.linalg.norm(w - vw) for vw in self.visited_weights])
            
            if dist_min > 0.15:
                if improvement_potential > max_improvement:
                    max_improvement = improvement_potential
                    best_w = w
                    
        return best_w

def evaluate_5d_agent(env, model, num_episodes=5):
    """Evalúa el agente y retorna el vector de recompensa promedio acumulado"""
    total_vec = np.zeros(5)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_vec = np.zeros(5)
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            if "vector_reward" in info:
                ep_vec += info["vector_reward"]
        total_vec += ep_vec
    return total_vec / num_episodes

def run_ols_final():
    env = gym.make("LunarLander-v3")
    env = Lander5ObjWrapper(env, np.ones(5)/5.0)
    
    ols = OLS(num_objectives=5)
    total_iterations = 9  # 5 esquinas + 4 refinamientos
    
    for iteration in range(total_iterations):
        
        target_weights = ols.get_next_weight()
        
        if target_weights is None:
            print("OLS Convergido: No se encontraron candidatos válidos.")
            break
            
        print(f"\n=== ITERACIÓN {iteration+1}/{total_iterations} ===")
        print(f"Pesos Objetivo: {np.round(target_weights, 3)}")
        print(f"Leyenda: [Centrado, Tiempo, Descenso, Estabilidad, Final]")

        env.update_weights(target_weights)
        
        # Entrenamiento con PPO
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, ent_coef=0.01)
        model.learn(total_timesteps=100000)
        
        # Evaluación del vector de desempeño sin escalarización
        val_vector = evaluate_5d_agent(env, model)
        
        filename = f"ols_fixed_iter_{iteration}"
        model.save(filename)
        ols.add_solution(target_weights, val_vector, filename)

        # Generación de video
        print(f"Grabando video para iteración {iteration}...")
        try:
            video_env = gym.make("LunarLander-v3", render_mode="rgb_array")
            video_env = Lander5ObjWrapper(video_env, target_weights)
            
            v_obs, _ = video_env.reset()
            frames = []
            v_done = False
            v_trunc = False
            while not (v_done or v_trunc):
                frames.append(video_env.render())
                v_action, _ = model.predict(v_obs, deterministic=True)
                v_obs, _, v_done, v_trunc, _ = video_env.step(v_action)
                
            imageio.mimsave(f"video_ols_{iteration}.mp4", frames, fps=30)
            video_env.close()
        except Exception as e:
            print(f"Error grabando video: {e}")

    print("\n=== FRONTERA DE PARETO APROXIMADA (CCS) ===")
    print(f"{'PESOS (Prioridad)':<40} | {'VALOR (Vector de Desempeño)':<40}")
    for sol in ols.ccs:
        w_str = str(np.round(sol['weights'], 2))
        v_str = str(np.round(sol['value'], 2))
        print(f"{w_str:<40} | {v_str:<40}")

if __name__ == "__main__":
    run_ols_final()