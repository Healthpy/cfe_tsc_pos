import numpy as np
from deap import base, creator, tools, algorithms
from .base import BaseCounterfactual
from utils.utils import initialize_metrics, update_metrics, save_metrics_to_csv, get_target_class
import time

class TSEvoCF(BaseCounterfactual):
    """TSEvoCF: Time Series Evolutionary Counterfactual Generation."""
    
    def __init__(self, model, data_name=None, pop_size=50, n_generations=100, mutation_prob=0.2):
        super().__init__(model)
        self.data_name = data_name
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_prob = mutation_prob
        
        # Clear any existing creators to avoid conflicts
        if 'FitnessMulti' in creator.__dict__:
            del creator.FitnessMulti
        if 'Individual' in creator.__dict__:
            del creator.Individual
            
        # Setup evolutionary components with proper weights
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
        
    def _setup_evolution(self, x_shape):
        """Configure evolutionary operators for given shape."""
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                            self.toolbox.attr_float, n=np.prod(x_shape))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=self.mutation_prob)
        self.toolbox.register("select", tools.selNSGA2)
        
    def _evaluate_individual(self, individual, x_original, target_class):
        """Evaluate fitness of an individual."""
        try:
            # Reshape the individual to match original shape
            cf = np.array(individual).reshape(x_original.shape)
            
            # Initialize fitness values
            distance = 1e6
            sparsity = 1e6
            
            # Only compute objectives if counterfactual is valid
            if self._is_valid_cf(cf, x_original, target_class):
                distance = float(np.sum(np.abs(cf - x_original)))
                sparsity = float(np.sum(np.abs(cf - x_original) > 1e-6))
            
            return distance, sparsity
            
        except Exception as e:
            print(f"Evaluation error: {str(e)}")
            return 1e6, 1e6

    def generate(self, x, target_class=None, X_train=None, y_train=None):
        """Generate counterfactual using evolutionary optimization."""
        if len(x.shape) == 3:
            x = x.reshape(x.shape[1], x.shape[2])
            
        if target_class is None:
            target_class = self._get_target_class(x)
            
        try:
            # Setup evolution for current instance
            self._setup_evolution(x.shape)
            
            # Setup evaluation function
            self.toolbox.register("evaluate", self._evaluate_individual, 
                                x_original=x, target_class=target_class)
            
            # Initialize population
            pop = self.toolbox.population(n=self.pop_size)
            
            # Evaluate initial population
            for ind in pop:
                ind.fitness.values = self.toolbox.evaluate(ind)
            
            # Evolution loop
            for gen in range(self.n_generations):
                # Select next generation
                offspring = algorithms.varOr(pop, self.toolbox, 
                                          lambda_=self.pop_size,
                                          cxpb=0.7, mutpb=0.3)
                
                # Evaluate new individuals
                for ind in offspring:
                    if not ind.fitness.valid:
                        ind.fitness.values = self.toolbox.evaluate(ind)
                
                # Select next generation
                pop = self.toolbox.select(pop + offspring, k=self.pop_size)
                
                # Check best solution
                best = tools.selBest(pop, k=1)[0]
                if best.fitness.valid and best.fitness.values[0] < 1e6:
                    cf = np.array(best).reshape(x.shape)
                    if self._is_valid_cf(cf, x, target_class):
                        return cf.reshape(1, cf.shape[0], cf.shape[1])
            
            return None
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return None