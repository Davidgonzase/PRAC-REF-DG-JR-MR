import gymnasium as gym
import imageio
import numpy as np
from stable_baselines3 import PPO

# -----------------------------------------------------------
# 1. WRAPPER DE 5 OBJETIVOS (CORREGIDO Y RE-ESCALADO)
# -----------------------------------------------------------
class Lander5ObjWrapper(gym.Wrapper):
    def __init__(self, env, current_weights):
        super().__init__(env)
        self.pasos = 0 
        self.contador_suelo = 0
        # Pesos iniciales
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
        
        # Lógica de temporizador (2 segundos / 100 frames)
        if esta_estable:
            self.contador_suelo += 1
        else:
            self.contador_suelo = 0 
        objetivo_cumplido = self.contador_suelo >= 100

        # --- VECTOR DE 5 DIMENSIONES ---
        # [0: Centrado, 1: Tiempo, 2: Descenso, 3: Estabilidad, 4: Aterrizaje]
        r_vec = np.zeros(5)

        if not objetivo_cumplido:
            
            # --- OBJ 0: CENTRADO (Navegación Lateral) ---
            # Penalización Densa. Escalado: 0.01 por paso.
            # Si se mantiene descentrado 300 pasos -> -3.0 acumulado (Comparable a aterrizaje)
            if not contacto_izq and not contacto_der:
                r_vec[0] = -abs(pos_x) * 0.05 

            # --- OBJ 2: CONTROL DE DESCENSO ---
            # Castigo por flotar
            if abs(pos_x) < 0.2 and vel_y > -0.2:
                r_vec[2] = -0.05
                
            # Premio denso por bajar suavemente
            # Escalado: 0.01 por paso. 100 pasos de bajada = +1.0 total.
            if -1.5 < vel_y < -0.5:
                r_vec[2] += 0.01

            # --- OBJ 1: EFICIENCIA TEMPORAL ---
            # CORRECCIÓN IMPORTANTE: Penalización constante pequeña.
            # -0.002 * 500 pasos = -1.0 (Equivalente a perder el bonus de aterrizaje)
            if self.contador_suelo == 0:
                r_vec[1] = -0.002
            
            # --- OBJ 3: ESTABILIDAD (Paciencia y Disciplina) ---
            if esta_estable:
                r_vec[3] = 0.02 # Premio por esperar
                
                # Penalización por "Disciplina" (tocar controles)
                # CORRECCIÓN: Reducido de -1.0 a -0.05 para no "matar" el aprendizaje
                if action != 0:
                     r_vec[3] -= 0.05
        
        else:
            # --- OBJ 4: ATERRIZAJE FINAL ---
            terminated = True
            # Recompensas Esparsas (Solo ocurren una vez, magnitud grande OK)
            if abs(pos_x) < 0.1:
                r_vec[4] = 1.0   # Éxito Total
            elif abs(pos_x) < 0.2:
                r_vec[4] = 0.5   # Éxito Parcial
            else:
                r_vec[4] = -0.5  # Aterrizaje Sucio

        # Penalización global por Crash
        if terminated and not objetivo_cumplido:
            # Si se estrella, penalizamos el objetivo final fuertemente
            if not esta_estable:
                r_vec[4] -= 1.0

        # Escalarización (Producto Punto: w * r)
        scalar_reward = np.dot(r_vec, self.current_weights)
        
        # Guardamos el vector puro
        info["vector_reward"] = r_vec
        
        return obs, scalar_reward, terminated, truncated, info

# -----------------------------------------------------------
# 2. SOLUCIONADOR OLS (MEJORADO CON PUNTO UTÓPICO)
# -----------------------------------------------------------
class HighDimOLSSolver:
    def __init__(self, num_objectives=5):
        self.num_obj = num_objectives
        self.ccs = [] # Convex Coverage Set (Soluciones encontradas)
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
        # FASE 1: Explorar esquinas (Objetivos puros)
        if len(self.ccs) < self.num_obj:
            next_w = np.zeros(self.num_obj)
            next_w[len(self.ccs)] = 1.0
            return next_w

        # FASE 2: Búsqueda OLS (Optimistic Linear Support)
        print("[OLS] Calculando siguiente peso óptimo (Maximin Improvement)...")
        
        best_w = None
        max_improvement = -np.inf
        
        # Punto Utópico Virtual: Asumimos que es posible obtener 1.0 en todo acumulado
        # (Ajustable según el entorno, pero 1.0 es una buena base normalizada)
        utopian_point = np.ones(self.num_obj) * 1.5 
        
        # Muestreo Monte Carlo
        for _ in range(5000): 
            w = np.random.rand(self.num_obj)
            w /= np.sum(w) # Normalizar suma a 1
            
            # 1. ¿Qué valor obtenemos AHORA con la mejor política existente para este w?
            current_max_val = -np.inf
            for sol in self.ccs:
                val = np.dot(sol["value"], w)
                if val > current_max_val:
                    current_max_val = val
            
            # 2. ¿Qué valor obtendríamos en el MEJOR caso teórico (Utópico)?
            optimistic_val = np.dot(utopian_point, w)
            
            # 3. La diferencia es la "posible mejora" (Regret)
            improvement_potential = optimistic_val - current_max_val
            
            # 4. Filtro de diversidad: No repetir pesos cercanos
            dist_min = min([np.linalg.norm(w - vw) for vw in self.visited_weights])
            
            if dist_min > 0.15: # Distancia mínima de seguridad en el espacio de pesos
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

# -----------------------------------------------------------
# 3. EJECUCIÓN DEL BUCLE OLS
# -----------------------------------------------------------
def run_ols_final():
    # Crear entorno base
    env = gym.make("LunarLander-v3")
    # Inicializamos wrapper con pesos dummy
    env = Lander5ObjWrapper(env, np.ones(5)/5.0)
    
    ols = HighDimOLSSolver(num_objectives=5)
    
    # 5 Esquinas + 4 Refinamientos
    total_iterations = 9 
    
    for iteration in range(total_iterations):
        
        target_weights = ols.get_next_weight()
        
        if target_weights is None:
            print("OLS Convergido: No se encontraron candidatos válidos.")
            break
            
        print(f"\n=== ITERACIÓN {iteration+1}/{total_iterations} ===")
        print(f"Pesos Objetivo: {np.round(target_weights, 3)}")
        print(f"Leyenda: [Centrado, Tiempo, Descenso, Estabilidad, Final]")

        # Actualizar pesos en el entorno
        env.update_weights(target_weights)
        
        # Entrenar
        # TIMESTEPS: Aumentado a 100k para dar tiempo a converger con las nuevas escalas
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, ent_coef=0.01)
        model.learn(total_timesteps=100000)
        
        # Evaluar el vector de desempeño real (sin escalarización)
        val_vector = evaluate_5d_agent(env, model)
        
        # Guardar solución en OLS
        filename = f"ols_fixed_iter_{iteration}"
        model.save(filename)
        ols.add_solution(target_weights, val_vector, filename)

        # Generar video de control
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