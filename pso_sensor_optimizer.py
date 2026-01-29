"""
Particle Swarm Optimization (PSO) para optimizar la ubicación de sensores DHT22
en un área de 1500 hectáreas, considerando zonas de sombra.

Autor: AI Sensors Project
Fecha: 2026-01-29
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


@dataclass
class PSOConfig:
    """Configuración de parámetros del algoritmo PSO."""
    population_size: int = 30  # M: Tamaño de la población
    omega: float = 0.9  # ω: Coeficiente de inercia inicial
    omega_min: float = 0.4  # Valor mínimo de inercia
    c1: float = 2.0  # C1: Parámetro cognitivo (atracción hacia mejor personal)
    c2: float = 2.0  # C2: Parámetro social (atracción hacia mejor global)
    max_iterations: int = 100  # Criterio de terminación: número máximo de iteraciones
    velocity_limit: float = 50.0  # Límite de velocidad máxima
    shadow_zone_penalty: float = 0.25  # 25% zona de sombra
    fitness_threshold: float = 0.001  # Umbral de convergencia
    
    # Dimensiones del área en metros (1500 hectáreas = 15,000,000 m²)
    # Asumiendo un área cuadrada: sqrt(15,000,000) ≈ 3873 m
    area_width: float = 3873.0  # metros
    area_height: float = 3873.0  # metros


@dataclass
class Particle:
    """Representa una partícula en el enjambre."""
    position: np.ndarray  # [x, y] posición actual
    velocity: np.ndarray  # [vx, vy] velocidad actual
    p_best_position: np.ndarray  # Mejor posición individual
    p_best_fitness: float  # Mejor fitness individual
    current_fitness: float = 0.0


class ShadowZoneMap:
    """
    Mapa de zonas de sombra para evaluar viabilidad solar.
    El 25% del terreno tiene sombra (sin viabilidad solar).
    """
    
    def __init__(self, width: float, height: float, shadow_ratio: float = 0.25, seed: int = 42):
        """
        Inicializa el mapa de zonas de sombra.
        
        Args:
            width: Ancho del área
            height: Alto del área
            shadow_ratio: Proporción del terreno con sombra (0.25 = 25%)
            seed: Semilla para reproducibilidad
        """
        self.width = width
        self.height = height
        self.shadow_ratio = shadow_ratio
        
        # Generar zonas de sombra como círculos aleatorios
        np.random.seed(seed)
        self.shadow_zones = self._generate_shadow_zones()
    
    def _generate_shadow_zones(self) -> List[Tuple[float, float, float]]:
        """Genera zonas de sombra como círculos (x, y, radio)."""
        zones = []
        total_area = self.width * self.height
        target_shadow_area = total_area * self.shadow_ratio
        current_shadow_area = 0
        
        while current_shadow_area < target_shadow_area:
            x = np.random.uniform(0, self.width)
            y = np.random.uniform(0, self.height)
            # Radios entre 50 y 200 metros
            radius = np.random.uniform(50, 200)
            zones.append((x, y, radius))
            current_shadow_area += np.pi * radius ** 2
        
        return zones
    
    def is_in_shadow(self, x: float, y: float) -> bool:
        """Verifica si una coordenada está en zona de sombra."""
        for zx, zy, radius in self.shadow_zones:
            distance = np.sqrt((x - zx) ** 2 + (y - zy) ** 2)
            if distance <= radius:
                return True
        return False
    
    def get_shadow_penalty(self, x: float, y: float) -> float:
        """
        Retorna un factor de penalización para posiciones en sombra.
        
        Returns:
            1.0 si está en sombra (máxima penalización)
            0.0 si no está en sombra (sin penalización)
        """
        return 1.0 if self.is_in_shadow(x, y) else 0.0


class FitnessEvaluator:
    """
    Evaluador de aptitud (fitness) para posiciones de sensores.
    
    Criterios de optimización:
    - Maximizar cobertura del área
    - Evitar zonas de sombra (penalización)
    - Distribución uniforme de sensores
    """
    
    def __init__(self, config: PSOConfig, shadow_map: ShadowZoneMap):
        self.config = config
        self.shadow_map = shadow_map
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evalúa la aptitud de una posición para un sensor.
        
        Args:
            position: Array [x, y] con las coordenadas
            
        Returns:
            Valor de fitness (mayor es mejor)
        """
        x, y = position
        
        # Verificar límites del área
        if not self._is_within_bounds(x, y):
            return -1000.0  # Penalización severa por estar fuera del área
        
        # Calcular fitness base (distancia al centro - favorece distribución)
        center_x = self.config.area_width / 2
        center_y = self.config.area_height / 2
        
        # Componente de cobertura: favorece posiciones que cubran el área
        coverage_score = self._calculate_coverage_score(x, y)
        
        # Penalización por zona de sombra (NOTA IMPORTANTE del pseudocódigo)
        shadow_penalty = self.shadow_map.get_shadow_penalty(x, y)
        
        if shadow_penalty > 0:
            # Si está en zona de sombra, descartar (fitness muy bajo)
            return -500.0 * shadow_penalty
        
        # Componente de irradiación solar simulada
        solar_score = self._calculate_solar_score(x, y)
        
        # Fitness combinado
        fitness = coverage_score + solar_score
        
        return fitness
    
    def _is_within_bounds(self, x: float, y: float) -> bool:
        """Verifica si la posición está dentro del área permitida."""
        return (0 <= x <= self.config.area_width and 
                0 <= y <= self.config.area_height)
    
    def _calculate_coverage_score(self, x: float, y: float) -> float:
        """
        Calcula un puntaje de cobertura basado en la posición.
        Favorece distribución uniforme desde el centro.
        """
        center_x = self.config.area_width / 2
        center_y = self.config.area_height / 2
        
        # Distancia normalizada al centro
        max_dist = np.sqrt(center_x**2 + center_y**2)
        dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Puntaje inverso a la distancia (posiciones más centradas son mejores)
        # pero también valoramos diversidad
        coverage = 100 * (1 - (dist_to_center / max_dist) ** 0.5)
        
        return coverage
    
    def _calculate_solar_score(self, x: float, y: float) -> float:
        """
        Simula un gradiente de irradiación solar.
        En este ejemplo, asumimos mejor irradiación en el lado sur (y bajo).
        """
        # Normalizar y
        normalized_y = y / self.config.area_height
        
        # Gradiente que favorece posiciones con y más bajo (sur)
        solar_score = 50 * (1 - normalized_y * 0.3)
        
        return solar_score


class ParticleSwarmOptimizer:
    """
    Implementación del algoritmo Particle Swarm Optimization
    para optimización de ubicación de sensores DHT22.
    """
    
    def __init__(self, config: PSOConfig, num_sensors: int = 1, seed: int = 42):
        """
        Inicializa el optimizador PSO.
        
        Args:
            config: Configuración de parámetros PSO
            num_sensors: Número de sensores a ubicar
            seed: Semilla para reproducibilidad
        """
        self.config = config
        self.num_sensors = num_sensors
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Crear mapa de sombras
        self.shadow_map = ShadowZoneMap(
            config.area_width, 
            config.area_height,
            config.shadow_zone_penalty,
            seed
        )
        
        # Evaluador de fitness
        self.fitness_evaluator = FitnessEvaluator(config, self.shadow_map)
        
        # Estado del enjambre
        self.particles: List[Particle] = []
        self.g_best_position: Optional[np.ndarray] = None
        self.g_best_fitness: float = float('-inf')
        
        # Historial para análisis
        self.fitness_history: List[float] = []
        self.convergence_history: List[np.ndarray] = []
    
    def initialize(self):
        """
        Paso 2: Inicialización del enjambre.
        - Genera M partículas con posiciones aleatorias
        - Inicializa velocidades aleatorias
        - Establece P_best inicial
        - Identifica G_best inicial
        """
        print("=" * 60)
        print("INICIALIZACIÓN DEL ENJAMBRE PSO")
        print("=" * 60)
        
        self.particles = []
        
        for j in range(self.config.population_size):
            # Generar posición aleatoria dentro del área
            position = np.array([
                np.random.uniform(0, self.config.area_width),
                np.random.uniform(0, self.config.area_height)
            ])
            
            # Inicializar velocidad aleatoria
            velocity = np.array([
                np.random.uniform(-self.config.velocity_limit, self.config.velocity_limit),
                np.random.uniform(-self.config.velocity_limit, self.config.velocity_limit)
            ])
            
            # Evaluar fitness inicial
            fitness = self.fitness_evaluator.evaluate(position)
            
            # Crear partícula con P_best = posición actual
            particle = Particle(
                position=position.copy(),
                velocity=velocity.copy(),
                p_best_position=position.copy(),
                p_best_fitness=fitness,
                current_fitness=fitness
            )
            
            self.particles.append(particle)
            
            # Actualizar G_best si esta partícula es mejor
            if fitness > self.g_best_fitness:
                self.g_best_fitness = fitness
                self.g_best_position = position.copy()
        
        print(f"✓ Creadas {self.config.population_size} partículas")
        print(f"✓ G_best inicial: posición={self.g_best_position}, fitness={self.g_best_fitness:.4f}")
        print()
    
    def update_velocity(self, particle: Particle, omega: float) -> np.ndarray:
        """
        Actualiza la velocidad de una partícula según la fórmula PSO:
        v_j,i(nuevo) = ω·v_j,i + C1·Rand()·(p_j,i - x_j,i) + C2·Rand()·(g_i - x_j,i)
        
        Args:
            particle: Partícula a actualizar
            omega: Coeficiente de inercia actual
            
        Returns:
            Nueva velocidad
        """
        r1 = np.random.random(2)  # Números aleatorios para componente cognitivo
        r2 = np.random.random(2)  # Números aleatorios para componente social
        
        # Componente de inercia
        inertia = omega * particle.velocity
        
        # Componente cognitivo (atracción hacia P_best)
        cognitive = self.config.c1 * r1 * (particle.p_best_position - particle.position)
        
        # Componente social (atracción hacia G_best)
        social = self.config.c2 * r2 * (self.g_best_position - particle.position)
        
        # Nueva velocidad
        new_velocity = inertia + cognitive + social
        
        return new_velocity
    
    def limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Control de velocidad: asegura que no exceda los límites permitidos.
        """
        magnitude = np.linalg.norm(velocity)
        if magnitude > self.config.velocity_limit:
            velocity = velocity * (self.config.velocity_limit / magnitude)
        return velocity
    
    def update_position(self, particle: Particle, new_velocity: np.ndarray) -> np.ndarray:
        """
        Actualiza la posición de la partícula:
        x_j,i(nuevo) = x_j,i + v_j,i(nuevo)
        """
        new_position = particle.position + new_velocity
        
        # Mantener dentro de los límites del área
        new_position[0] = np.clip(new_position[0], 0, self.config.area_width)
        new_position[1] = np.clip(new_position[1], 0, self.config.area_height)
        
        return new_position
    
    def update_inertia(self, iteration: int) -> float:
        """
        Actualiza el parámetro ω (reducción lineal para refinar la búsqueda).
        """
        omega_range = self.config.omega - self.config.omega_min
        omega = self.config.omega - (omega_range * iteration / self.config.max_iterations)
        return max(omega, self.config.omega_min)
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Paso 3: Ciclo principal de optimización.
        
        Args:
            verbose: Si True, muestra progreso
            
        Returns:
            Tuple con (mejor posición global, mejor fitness)
        """
        print("=" * 60)
        print("CICLO PRINCIPAL DE OPTIMIZACIÓN PSO")
        print("=" * 60)
        
        # Inicializar si no se ha hecho
        if not self.particles:
            self.initialize()
        
        iteration = 0
        converged = False
        previous_best = self.g_best_fitness
        
        while iteration < self.config.max_iterations and not converged:
            omega = self.update_inertia(iteration)
            
            # Para cada partícula en el enjambre
            for particle in self.particles:
                # 1. Calcular nueva velocidad
                new_velocity = self.update_velocity(particle, omega)
                
                # 2. Control de velocidad
                new_velocity = self.limit_velocity(new_velocity)
                
                # 3. Actualizar posición
                new_position = self.update_position(particle, new_velocity)
                
                # 4. Evaluar aptitud (fitness) con penalización de sombra
                fitness = self.fitness_evaluator.evaluate(new_position)
                
                # Actualizar estado de la partícula
                particle.velocity = new_velocity
                particle.position = new_position
                particle.current_fitness = fitness
                
                # 5. Actualizar P_best si el fitness actual es mejor
                if fitness > particle.p_best_fitness:
                    particle.p_best_fitness = fitness
                    particle.p_best_position = new_position.copy()
            
            # 6. Actualizar G_best
            for particle in self.particles:
                if particle.p_best_fitness > self.g_best_fitness:
                    self.g_best_fitness = particle.p_best_fitness
                    self.g_best_position = particle.p_best_position.copy()
            
            # Guardar historial
            self.fitness_history.append(self.g_best_fitness)
            self.convergence_history.append(self.g_best_position.copy())
            
            # Verificar convergencia
            if abs(self.g_best_fitness - previous_best) < self.config.fitness_threshold:
                converged = True
            previous_best = self.g_best_fitness
            
            # Mostrar progreso
            if verbose and (iteration % 10 == 0 or iteration == self.config.max_iterations - 1):
                print(f"Iteración {iteration:3d} | ω={omega:.4f} | "
                      f"G_best fitness={self.g_best_fitness:.4f} | "
                      f"Posición: ({self.g_best_position[0]:.1f}, {self.g_best_position[1]:.1f})")
            
            iteration += 1
        
        print()
        if converged:
            print(f"✓ Convergencia alcanzada en iteración {iteration}")
        else:
            print(f"✓ Máximo de iteraciones alcanzado ({self.config.max_iterations})")
        
        return self.g_best_position, self.g_best_fitness
    
    def get_results(self) -> dict:
        """
        Paso 5: Reportar G_best como la configuración óptima.
        """
        return {
            'optimal_position': self.g_best_position,
            'optimal_fitness': self.g_best_fitness,
            'iterations': len(self.fitness_history),
            'final_omega': self.update_inertia(len(self.fitness_history)),
            'in_shadow_zone': self.shadow_map.is_in_shadow(
                self.g_best_position[0], 
                self.g_best_position[1]
            ) if self.g_best_position is not None else None
        }
    
    def visualize(self, save_path: Optional[str] = None):
        """
        Visualiza el resultado de la optimización.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Mapa del área con posiciones finales de partículas
        ax1 = axes[0]
        
        # Dibujar zonas de sombra
        for zx, zy, radius in self.shadow_map.shadow_zones:
            circle = plt.Circle((zx, zy), radius, color='gray', alpha=0.3)
            ax1.add_patch(circle)
        
        # Dibujar posiciones finales de partículas
        positions = np.array([p.position for p in self.particles])
        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', alpha=0.6, 
                   label='Partículas finales', s=50)
        
        # Dibujar mejor posición global
        if self.g_best_position is not None:
            ax1.scatter(self.g_best_position[0], self.g_best_position[1], 
                       c='red', marker='*', s=300, label='G_best (Óptimo)', 
                       edgecolors='black', linewidth=2)
        
        ax1.set_xlim(0, self.config.area_width)
        ax1.set_ylim(0, self.config.area_height)
        ax1.set_xlabel('X (metros)')
        ax1.set_ylabel('Y (metros)')
        ax1.set_title('Ubicación Óptima de Sensor DHT22\n(Zonas grises = sombra)')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Evolución del fitness
        ax2 = axes[1]
        ax2.plot(self.fitness_history, 'b-', linewidth=2)
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Mejor Fitness Global')
        ax2.set_title('Convergencia del Algoritmo PSO')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Gráfico guardado en: {save_path}")
        
        plt.show()


def main():
    """
    Función principal para demostrar el algoritmo PSO.
    """
    print("=" * 60)
    print("OPTIMIZACIÓN PSO PARA UBICACIÓN DE SENSORES DHT22")
    print("Área: 1500 hectáreas | Zona de sombra: 25%")
    print("=" * 60)
    print()
    
    # Paso 1: Configurar parámetros
    config = PSOConfig(
        population_size=30,      # M: Tamaño del enjambre
        omega=0.9,               # ω: Coeficiente de inercia inicial
        omega_min=0.4,           # ω mínimo
        c1=2.0,                  # C1: Parámetro cognitivo
        c2=2.0,                  # C2: Parámetro social
        max_iterations=100,      # Criterio de terminación
        velocity_limit=100.0,    # Límite de velocidad
        shadow_zone_penalty=0.25 # 25% zona de sombra
    )
    
    print("PARÁMETROS CONFIGURADOS:")
    print(f"  • Tamaño de población (M): {config.population_size}")
    print(f"  • Coeficiente de inercia (ω): {config.omega} → {config.omega_min}")
    print(f"  • Parámetro cognitivo (C1): {config.c1}")
    print(f"  • Parámetro social (C2): {config.c2}")
    print(f"  • Máx. iteraciones: {config.max_iterations}")
    print(f"  • Límite de velocidad: {config.velocity_limit}")
    print(f"  • Zona de sombra: {config.shadow_zone_penalty * 100}%")
    print(f"  • Dimensiones del área: {config.area_width:.0f}m x {config.area_height:.0f}m")
    print()
    
    # Crear optimizador
    optimizer = ParticleSwarmOptimizer(config, seed=42)
    
    # Paso 2: Inicialización
    optimizer.initialize()
    
    # Paso 3: Ciclo principal de optimización
    best_position, best_fitness = optimizer.optimize(verbose=True)
    
    # Paso 5: Reportar resultado
    print()
    print("=" * 60)
    print("RESULTADO: CONFIGURACIÓN ÓPTIMA")
    print("=" * 60)
    
    results = optimizer.get_results()
    print(f"  Posición óptima (G_best):")
    print(f"    • X: {results['optimal_position'][0]:.2f} metros")
    print(f"    • Y: {results['optimal_position'][1]:.2f} metros")
    print(f"  Fitness óptimo: {results['optimal_fitness']:.4f}")
    print(f"  Iteraciones totales: {results['iterations']}")
    print(f"  En zona de sombra: {'Sí ⚠️' if results['in_shadow_zone'] else 'No ✓'}")
    print()
    
    # Visualizar resultados
    optimizer.visualize(save_path='pso_optimization_result.png')
    
    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()
