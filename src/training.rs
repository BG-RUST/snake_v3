use crate:: {
    game::Game,
    evolution::{Population, Individual },
    game_input::GameInput,
};

use std::time::Instant;

const POP_SIZE: usize = 50;
const GENERATIONS: usize = 5000;
const MAX_STEPS: usize = 10000;
const RETAIN_TOP: usize = 10;
const MUTATION_RATE: f32 = 0.05;
const MUTATION_MAG: f32 = 0.2;

pub fn run_training(width: usize, height: usize) {
    let mut population = Population::new_random(POP_SIZE);

    for generation in 0..GENERATIONS {
        let start_time = Instant::now();

        for ind in &mut population.individuals {
            let mut game = Game::new(width, height);
            let mut steps = 0;
            let mut steps_since_eat = 0;
            let start_pos = game.snake().head();

            while steps < MAX_STEPS {
                let input = GameInput::from_game(&game, 0, start_pos, steps_since_eat);
                let action = ind.genome.decide(&input);
                game.set_action_index(action);

                game.update();

                steps += 1;
                steps_since_eat += 1;

                if game.snake().head() == game.food() {
                    steps_since_eat = 0;
                }

                if game.snake().is_dead(game.width(), game.height()) {
                    break;
                }
            }

            let eaten = game.snake().body().len().saturating_sub(1);
            let fitness = eaten as f32 * 10.0 + steps as f32 * 0.01;

            ind.fitness = fitness;
            ind.steps = steps;
            ind.eaten = eaten;

        }

        let best = population.best();
        let avg_eaten: f32 = population.individuals.iter().map(|i| i.eaten).sum::<usize>() as f32 / POP_SIZE as f32;
        println!(
            "Gen {:>3} | Best Fit: {:>7.2} | Eaten: {:>2} | Steps: {:>4} | Avg Eaten: {:.2} | Time: {:?}",
            generation,
            best.fitness,
            best.eaten,
            best.steps,
            avg_eaten,
            start_time.elapsed()
        );

        population.evolve(RETAIN_TOP, MUTATION_RATE, MUTATION_MAG);
    }
}

fn show_best_individual(ind: Individual, width: usize, height: usize) {
    let mut game = Game::new(width, height);
    let mut steps = 0;
    let start_pos = game.snake().head();
    let mut steps_since_eat = 0;

    while steps < MAX_STEPS {
        std::thread::sleep(std::time::Duration::from_millis(100));

        let input = GameInput::from_game(&game, 0, start_pos, steps_since_eat);
        let action = ind.genome.decide(&input);
        game.set_action_index(action);
        game.update();

        println!("Step: {:>4} | Length: {} | Pos: {:?}", steps, game.snake().body().len(), game.snake().head());

        steps += 1;
        steps_since_eat += 1;

        if game.snake().head() == game.food() {
            steps_since_eat = 0;
        }

        if game.snake().is_dead(game.width(), game.height()) {
            println!("💀 DEAD at step {}", steps);
            break;
        }
    }

    println!("✅ Finished! Final length: {}", game.snake().body().len());
}