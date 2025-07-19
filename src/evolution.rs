use crate::genome::*;
use rand::Rng;
use crate::db as db_model;

#[derive(Clone)]
pub struct Individual {
    pub genome: Genome,
    pub fitness: f32,
    pub steps: usize,
    pub eaten: usize,
}

pub struct Population {
    pub individuals: Vec<Individual>,
    pub generation: u32,
}

impl Population {
    pub fn new_random(size: usize) -> Self {
        let individuals = (0..size)
            .map(|_| Individual {
                genome: Genome::random(),
                fitness: 0.0,
                steps: 0,
                eaten: 0,
            })
            .collect();

        Self {
            individuals,
            generation: 0,
        }
    }

    /// Создание популяции с использованием лучшего из базы, если есть
    pub fn from_best_or_random(conn: &rusqlite::Connection, size: usize) -> Self {
        let mut individuals = Vec::new();

        if let Some(best_db) = crate::db::get_best_individual(conn) {
            let best = Individual::from(best_db);
            individuals.push(best.clone());

            while individuals.len() < size {
                let mut new = best.clone();
                new.genome.mutate(0.1, 0.2);
                individuals.push(new);
            }

            println!("✅ Загружен лучший из базы и использован как стартовая точка");
        } else {
            println!("⚠️ В базе нет особей, начинаем с нуля");
            return Self::new_random(size);
        }

        Self {
            individuals,
            generation: 0,
        }
    }

    pub fn evolve(&mut self, retain_top: usize, mutation_rate: f32, mutation_mag: f32) {
        self.individuals
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut new_gen = self.individuals[..retain_top]
            .iter()
            .cloned()
            .collect::<Vec<_>>();


        let mut rng = rand::thread_rng();
        while new_gen.len() < self.individuals.len() {
            let parent1 = &self.individuals[rng.gen_range(0..retain_top)].genome;
            let parent2 = &self.individuals[rng.gen_range(0..retain_top)].genome;

            let mut child = Genome::crossover(parent1, parent2);
            child.mutate(mutation_rate, mutation_mag);

            new_gen.push(Individual {
                genome: child,
                fitness: 0.0,
                steps: 0,
                eaten: 0,
            });
        }

        self.individuals = new_gen;
        self.generation += 1;
    }

    pub fn best(&self) -> &Individual {
        self.individuals
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }
}

impl From<db_model::Individual> for Individual {
    fn from(db_ind: db_model::Individual) -> Self {
        Individual {
            genome: Genome {
                weights: db_ind.genome,
            },
            fitness: db_ind.fitness,
            steps: db_ind.steps,
            eaten: db_ind.eaten,
        }
    }
}
