use rand::Rng;
use std::sync::{Arc, Mutex};
use std::fs::{OpenOptions, write};
use std::io::Write;
use rayon::prelude::*;
use crate::net::*;
use crate::snake::*;
use crate::food::Food;

const POPULATION_SIZE: usize = 2000;
const BASE_MUTATION_RATE: f32 = 0.05;
const BASE_MUTATION_STRENGTH: f32 = 0.1;
const FOOD_REWARD: f32 = 5000.0;
const MAX_NO_FOOD_STEPS: u32 = 200;
const STAGNATION_LIMIT: u32 = 20;
//неработает так как надо
pub fn run_ga_training(best_net: Arc<Mutex<NeuralNet>>) {
    let initial_best = { best_net.lock().unwrap().clone() };
    let input_size = initial_best.input_size;
    let hidden_size = initial_best.hidden_size;
    let output_size = initial_best.output_size;

    let mut population: Vec<NeuralNet> = Vec::with_capacity(POPULATION_SIZE);
    population.push(initial_best);
    for _ in 1..POPULATION_SIZE {
        population.push(NeuralNet::new(input_size, hidden_size, output_size));
    }

    let mut generation: u32 = 0;
    let mut rng = rand::thread_rng();

    let mut best_fitness_ever: f32 = std::fs::read_to_string("best_fitness.txt")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .unwrap_or(f32::MIN);

    let mut mutation_rate = BASE_MUTATION_RATE;
    let mut mutation_strength = BASE_MUTATION_STRENGTH;
    let mut stagnation_counter = 0;

    println!("Запуск генетического алгоритма с популяцией {}...", POPULATION_SIZE);

    let mut log_file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("fitness_log.csv")
        .expect("Не удалось открыть файл логов");

    loop {
        generation += 1;

        let fitnesses: Vec<f32> = population
            .par_iter()
            .map(|net| evaluate_network(net))
            .collect();

        let (best_index, best_fitness) = fitnesses
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let total_fitness: f32 = fitnesses.iter().sum();
        let average_fitness = total_fitness / POPULATION_SIZE as f32;

        writeln!(log_file, "{},{},{}", generation, average_fitness, best_fitness).unwrap();

        let best_network = population[best_index].clone();
        if *best_fitness > best_fitness_ever {
            best_fitness_ever = *best_fitness;
            stagnation_counter = 0;
            {
                let mut global_best = best_net.lock().unwrap();
                *global_best = best_network.clone();
            }
            best_network.save_to_file("best_weights.txt");
            write("best_fitness.txt", format!("{}", best_fitness_ever)).unwrap();
            println!("Поколение {}: Новый лучший фитнес = {:.2}. Сохраняем веса в файл.", generation, best_fitness);
            mutation_rate = BASE_MUTATION_RATE;
            mutation_strength = BASE_MUTATION_STRENGTH;
        } else {
            stagnation_counter += 1;
            if stagnation_counter >= STAGNATION_LIMIT {
                mutation_rate = 0.2;
                mutation_strength = 0.3;
                println!("\u{26a0}\u{fe0f} Стагнация {} поколений — усиливаем мутацию!", stagnation_counter);
            }
            println!("Поколение {}: Лучший фитнес = {:.2}, Средний фитнес = {:.2}", generation, best_fitness, average_fitness);
        }

        let new_population: Vec<NeuralNet> = (0..POPULATION_SIZE - 1)
            .into_par_iter()
            .map(|_| {
                let mut thread_rng = rand::thread_rng();
                let p1 = select_parent(&fitnesses, &mut thread_rng);
                let mut p2 = select_parent(&fitnesses, &mut thread_rng);
                while p1 == p2 {
                    p2 = select_parent(&fitnesses, &mut thread_rng);
                }
                let parent1 = &population[p1];
                let parent2 = &population[p2];

                let mut child = crossover(parent1, parent2);
                mutate(&mut child, &mut thread_rng, mutation_rate, mutation_strength);
                child
            })
            .collect();

        let mut new_population = new_population;
        new_population.push(population[best_index].clone());
        population = new_population;
    }
}

fn select_parent(fitnesses: &Vec<f32>, rng: &mut impl rand::Rng) -> usize {
    let tournament_size = 5;
    let mut best_idx = rng.gen_range(0..fitnesses.len());
    let mut best_fit = fitnesses[best_idx];

    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..fitnesses.len());
        if fitnesses[idx] > best_fit {
            best_fit = fitnesses[idx];
            best_idx = idx;
        }
    }

    best_idx
}

fn evaluate_network(net: &NeuralNet) -> f32 {
    let mut snake = Snake::new();
    let mut food = Food::new_safe(800, 600, 20, &snake);

    let mut fitness = 0.0;
    let mut steps_since_food = 0;
    let mut prev_dist = manhattan(snake.body[0], food.position);
    use std::collections::HashSet;
    let mut visited = HashSet::new();

    loop {
        visited.insert(snake.body[0]);
        let dir = net.decide_direction(&snake, &food, 800, 600);
        snake.change_dir(dir);
        snake.update();

        let head = snake.body[0];
        let dist = manhattan(head, food.position);

        if dist < prev_dist {
            fitness += 1.0;
        } else {
            fitness -= 0.5;
        }
        prev_dist = dist;

        if head == food.position {
            fitness += FOOD_REWARD;
            snake.grow();
            food = Food::new_safe(800, 600, 20, &snake);
            prev_dist = manhattan(snake.body[0], food.position);
            steps_since_food = 0;
        } else {
            steps_since_food += 1;
        }

        fitness += 0.01;
        if steps_since_food > 200 {
            fitness -= 0.1 * (steps_since_food as f32 / 200.0);
        }
        if visited.len() < (snake.body.len() + 5) {
            fitness -= 0.05;
        }

        if is_hit_wall(head, 800, 600, 20)
            || snake.is_colliding_with_self()
            || steps_since_food > MAX_NO_FOOD_STEPS
            || fitness.is_nan() {
            break;
        }
    }

    fitness
}

fn mutate(net: &mut NeuralNet, rng: &mut rand::rngs::ThreadRng, mutation_rate: f32, mutation_strength: f32) {
    for i in 0..net.input_size {
        for j in 0..net.hidden_size {
            if rng.r#gen::<f32>() < mutation_rate {
                let delta = rng.gen_range(-mutation_strength..mutation_strength);
                net.weights_input_hidden[i][j] += delta;
            }
        }
    }
    for j in 0..net.hidden_size {
        if rng.r#gen::<f32>() < mutation_rate {
            let delta = rng.gen_range(-mutation_strength..mutation_strength);
            net.bias_hidden[j] += delta;
        }
    }
    for j in 0..net.hidden_size {
        for k in 0..net.output_size {
            if rng.r#gen::<f32>() < mutation_rate {
                let delta = rng.gen_range(-mutation_strength..mutation_strength);
                net.weights_hidden_output[j][k] += delta;
            }
        }
    }
    for k in 0..net.output_size {
        if rng.r#gen::<f32>() < mutation_rate {
            let delta = rng.gen_range(-mutation_strength..mutation_strength);
            net.bias_output[k] += delta;
        }
    }
}

fn crossover(parent1: &NeuralNet, parent2: &NeuralNet) -> NeuralNet {
    let mut child = parent1.clone();
    for i in 0..parent1.input_size {
        for j in 0..parent1.hidden_size {
            if rand::random::<f32>() < 0.5 {
                child.weights_input_hidden[i][j] = parent2.weights_input_hidden[i][j];
            }
        }
    }
    for j in 0..parent1.hidden_size {
        if rand::random::<f32>() < 0.5 {
            child.bias_hidden[j] = parent2.bias_hidden[j];
        }
    }
    for j in 0..parent1.hidden_size {
        for k in 0..parent1.output_size {
            if rand::random::<f32>() < 0.5 {
                child.weights_hidden_output[j][k] = parent2.weights_hidden_output[j][k];
            }
        }
    }
    for k in 0..parent1.output_size {
        if rand::random::<f32>() < 0.5 {
            child.bias_output[k] = parent2.bias_output[k];
        }
    }
    child
}

fn manhattan(a: (u32, u32), b: (u32, u32)) -> f32 {
    ((a.0 as i32 - b.0 as i32).abs() + (a.1 as i32 - b.1 as i32).abs()) as f32
}

fn is_hit_wall(head: (u32, u32), width: u32, height: u32, cell_size: u32) -> bool {
    let cols = width / cell_size;
    let rows = height / cell_size;
    let (x, y) = head;
    x <= 1 || y <= 1 || x >= cols - 1 || y >= rows - 1
}
