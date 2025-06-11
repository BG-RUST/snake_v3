use ga::Genome;
use neural::NeuralNet;
use game::Game;
use winit::event_loop::EventLoop;
use std::fs::OpenOptions;
use std::io::Write;

mod game;
mod snake;
mod food;
mod border;
mod neural;
mod ga;

fn main() {
    let mut event_loop = EventLoop::new();

    let population_size = 50;
    let generations = 100;

    let mut population: Vec<Genome> = if let Some(saved) = Genome::load_from_file("best_genome.json") {
        println!("🔄 Загружаем сохранённый genome");
        vec![saved.clone(); population_size]
    } else {
        println!("🆕 Старт с нуля");
        (0..population_size).map(|_| Genome::new_random()).collect()
    };


    for generation in 0..generations {
        println!("🧬 Поколение {}", generation + 1);

        for genome in &mut population {
            let net = genome.to_network();
            let game = Game::new(640, 640);
            let score = game.run_with_ai(&net, &mut event_loop);
            genome.fitness = (score * 100) as f32;
        }

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let best_fitness = population[0].fitness;
        let avg_fitness = population.iter().map(|g| g.fitness).sum::<f32>() / population.len() as f32;

        println!("Best result: {:.2}, Average: {:.2}", best_fitness, avg_fitness);

        log_generation(generation, best_fitness, avg_fitness);



        // selection of everything and creation of a new generation
        let elites = &population[0..5];
        let mut new_gen = elites.to_vec();

        while new_gen.len() < population_size {
            use rand::seq::SliceRandom;
            use rand::Rng;
            let mut rng = rand::thread_rng();

            let parent_a = elites.choose(&mut rng).unwrap();
            let parent_b = elites.choose(&mut rng).unwrap();

            let mut child_weights = parent_a.weights.iter().zip(&parent_b.weights)
                .map(|(a, b)| if rng.gen_bool(0.5) { *a } else { *b })
                .collect::<Vec<f32>>();

            //mutation
            for w in &mut child_weights {
                if rng.gen_bool(0.1) {
                    *w += rng.gen_range(-0.5..0.5);
                }
            }
            new_gen.push(Genome {weights: child_weights, fitness: 0.0, });
        }
        population = new_gen;


    }
}
//log csv

fn log_generation(generation: usize, best: f32, avg: f32) {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("train_log_csv.csv")
        .expect("Could not open file");

    if generation == 0 {
        writeln!(file, "generation,best,average").unwrap();
    }
    writeln!(file, "{},{},{}", generation + 1, best, avg).unwrap();

}