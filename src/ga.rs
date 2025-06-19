use rand::Rng;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use crate::net::*;
use crate::snake::*;
use crate::food::Food;

/// Параметры генетического алгоритма
const POPULATION_SIZE: usize = 100;      // размер популяции (число особей / нейросетей в поколении)
const MUTATION_RATE: f32 = 0.1;         // вероятность мутации веса (10%)
const MUTATION_STRENGTH: f32 = 0.5;     // максимальная величина случайного изменения веса при мутации
const FOOD_REWARD: f32 = 1000.0;        // бонус к фитнесу за съеденную еду
const STEP_REWARD: f32 = 1.0;           // награда за каждый шаг (можно оставить 1.0, чтобы учитывалось время жизни)
const MAX_NO_FOOD_STEPS: u32 = 200;     // максимальное количество шагов без еды до принудительного завершения (чтобы змейка не бесконечно блуждала)

// Структура для хранения результата (фитнеса) игры одной змейки
struct GameResult {
    steps: u32,
    food_eaten: u32,
}

/// Запускает цикл генетического алгоритма в фоновом потоке.
/// Постоянно улучшает нейросеть, обновляя `best_net` наилучшими найденными весами.

pub fn run_ga_training(best_net: Arc<Mutex<NeuralNet>>) {
    // Инициализируем популяцию нейросетей
    // Если у нас уже есть сохраненная лучшая сеть (best_net), используем ее как первую особь.
    let initial_best = {best_net.lock().unwrap().clone() };
    let input_size = initial_best.input_size;
    let hidden_size = initial_best.hidden_size;
    let output_size = initial_best.output_size;

    let mut population: Vec<NeuralNet> = Vec::with_capacity(POPULATION_SIZE);
    population.push(initial_best);
    // Остальные особи - новые случайные нейросети
    for _ in 1..POPULATION_SIZE {
        population.push(NeuralNet::new(input_size, hidden_size, output_size));
    }

    let mut generation: u32 = 0;
    let mut rng = rand::thread_rng();
    let mut best_fitness_ever: f32 = f32::MIN; // отслеживаем наилучший фитнес за все поколения
    println!("Запуск генетического алгоритма с популяцией {}...", POPULATION_SIZE);
    // Основной цикл поколений GA
    loop {
        generation += 1;
        // 1. Оцениваем каждую сеть (индивида) в текущей популяции, получаем их приспособленность (фитнес)
        let mut fitnesses: Vec<f32> = Vec::with_capacity(POPULATION_SIZE);
        let mut best_index = 0;
        let mut best_fitness = f32::MIN;
        let mut total_fitness = 0.0;
        for (index, net) in population.iter().enumerate() {
            let fitness = evaluate_network(net);
            fitnesses.push(fitness);
            total_fitness += fitness;
            // Находим индекс лучшей особи
            if fitness > best_fitness {
                best_fitness = fitness;
                best_index = index;
            }
        }
        let average_fitness = total_fitness / POPULATION_SIZE as f32;

        // Сохраняем лучшую сеть текущего поколения
        let best_network = population[best_index].clone();
        if best_fitness > best_fitness_ever {
            best_fitness_ever = best_fitness;
            // Обновляем глобальную лучшую сеть (для использования в игре)
            // и сохраняем ее веса в файл
            {
                let mut global_best = best_net.lock().unwrap();
                *global_best = best_network.clone();
            }
            best_network.save_to_file("best_weights.txt");
            println!("Поколение {}: Новый лучший фитнес = {:.2}. Сохраняем веса в файл.", generation, best_fitness);

        } else {
            println!("Поколение {}: Лучший фитнес = {:.2}, Средний фитнес = {:.2}", generation, best_fitness, average_fitness);

        }

        // 2. Отбор (Selection):
        // Реализуем отбор с помощью турнира: случайно выбираем несколько особей и берем лучшую из них.
        let tournament_size = 5;
        let mut select_parent = |rng: &mut rand::rngs::ThreadRng| -> usize {
            let mut best_idx = rng.gen_range(0..POPULATION_SIZE);
            let mut best_fit = fitnesses[best_idx];
            // выбираем tournament_size случайных индексов и находим среди них наилучший фитнес
            for _ in 1..tournament_size {
                let idx = rng.gen_range(0..POPULATION_SIZE);
                if fitnesses[idx] > best_fit {
                    best_fit = fitnesses[idx];
                    best_idx = idx;
                }
            }
            best_idx
        };

        // 3. Воспроизведение (Reproduction): формируем новое поколение.
        let mut new_population: Vec<NeuralNet> = Vec::with_capacity(POPULATION_SIZE);
        // Элитизм: переносим лучшую особь без изменений в следующее поколение
        new_population.push(population[best_index].clone());
        // Остальных особей генерируем путем скрещивания (crossover) и мутаций
        while new_population.len() < POPULATION_SIZE {
            // Выбираем двух родителей (их индексы в старой популяции) с помощью турнира
            let mut p1_index = select_parent(&mut rng);
            let mut p2_index = select_parent(&mut rng);
            // гарантируем, что родители разные (в редком случае, когда турнир вернул одинаковых)
            if p1_index == p2_index {
                p2_index = select_parent(&mut rng);
            }
            let parent1 = &population[p1_index];
            let parent2 = &population[p2_index];
            // Создаем нового ребенка путем скрещивания генов родителей
            let mut child = crossover(parent1, parent2);
            // Применяем мутацию к ребенку
            mutate(&mut child, &mut rng);
            // Добавляем ребенка в новую популяцию
            new_population.push(child);
        }
        // Обновляем популяцию новым поколением
        population = new_population;
        // (Опционально) можно сделать небольшую паузу между поколениями,
        // чтобы не нагружать CPU на 100%, и чтобы визуально успевать следить за выводом:
        // std::thread::sleep(Duration::from_millis(10));

    }

}
/// Функция оценки (fitness function): запускает игру для заданной нейросети `net` и вычисляет фитнес.
/// Фитнес основывается на количестве съеденной еды и пройденных шагов (чем больше, тем лучше).
fn evaluate_network(net: &NeuralNet) -> f32 {
    // Создаем отдельную змейку и еду для симуляции (не связаны с основной игрой)
    let mut snake = Snake::new();
    let mut food = Food::new(800, 600, 20);
    let mut steps = 0;
    let mut food_eaten = 0;
    let mut steps_since_last_food = 0;

    // Выполняем симуляцию игры для данной нейросети
    loop {
        steps += 1;
        steps_since_last_food += 1;
        // Нейросеть принимает решение, куда двигаться
        let direction = net.decide_direction(&snake, &food, 800, 600);
        snake.change_dir(direction);
        snake.update();

        // Проверяем, не съела ли змейка еду на этом шаге
        if snake.body[0] == food.position {
            food_eaten += 1;
            snake.grow();
            food = Food::new(800, 600, 20);
            steps_since_last_food = 0;
        }

        // Проверяем столкновения со стеной или с собой
        let (head_x, head_y) = snake.body[0];
        let cols = 800 / 20;
        let rows = 600 / 20;
        let hit_wall = head_x == 1 || head_y == 1 || head_x >= cols - 1 || head_y >= rows - 1;
        let hit_self = snake.is_colliding_with_self();
        // Проверяем условие истощения: слишком долго без пищи
        let starved = steps_since_last_food > MAX_NO_FOOD_STEPS;
        if hit_wall || hit_self || starved {
            break;
        }
        // Ограничение по шагам (предохранитель, чтобы игра не шла бесконечно)
        if steps > 10000 {
            break;
        }
    }
    // Рассчитываем фитнес:
    // количество шагов (время жизни) * награда за шаг + количество еды * большой бонус
    let fitness = steps as f32 * STEP_REWARD + food_eaten as f32 * FOOD_REWARD;
    fitness

}

/// Скрещивание (crossover) двух родительских нейросетей для создания ребенка.
/// Мы используем равновероятное (uniform) скрещивание: каждый вес ребенка берется от одного из родителей с равной вероятностью.
fn crossover(parent1: &NeuralNet, parent2: &NeuralNet) -> NeuralNet {
    let mut child = parent1.clone();
    // Скрещиваем веса input->hidden
    for i in 0..parent1.input_size {
        for j in 0..parent1.hidden_size {
            if rand::random::<f32>() < 0.5 {
                // с вероятностью 50% берем ген от второго родителя
                child.weights_input_hidden[i][j] = parent2.weights_input_hidden[i][j];
            }
        }
    }
    // Скрещиваем bias скрытого слоя
    for j in 0..parent1.hidden_size {
        if rand::random::<f32>() < 0.5 {
            child.bias_hidden[j] = parent2.bias_hidden[j];
        }
    }
    // Скрещиваем веса hidden->output
    for j in 0..parent1.hidden_size {
        for k in 0..parent1.output_size {
            if rand::random::<f32>() < 0.5 {
                child.weights_hidden_output[j][k] = parent2.weights_hidden_output[j][k];
            }
        }
    }
    // Скрещиваем bias выходного слоя
    for k in 0..parent1.output_size {
        if rand::random::<f32>() < 0.5 {
            child.bias_output[k] = parent2.bias_output[k];
        }
    }
    child
}


/// Мутация: случайно изменяет некоторые веса у данной нейросети `net`.
/// Для каждого веса и bias с вероятностью MUTATION_RATE происходит изменение на небольшое случайное значение.
fn mutate(net: &mut NeuralNet, rng: &mut rand::rngs::ThreadRng) {
    for i in 0..net.input_size {
        for j in 0..net.hidden_size {
            if rng.r#gen::<f32>() < MUTATION_RATE {
                let delta = rng.gen_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
                net.weights_input_hidden[i][j] += delta;
            }
        }
    }
    // Мутация bias скрытого слоя
    for j in 0..net.hidden_size {
        if rng.r#gen::<f32>() < MUTATION_RATE {
            let delta = rng.gen_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
            net.bias_hidden[j] += delta;
        }
    }
    // Мутация весов hidden->output
    for j in 0..net.hidden_size {
        for k in 0..net.output_size {
            if rng.r#gen::<f32>() < MUTATION_RATE {
                let delta = rng.gen_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
                net.weights_hidden_output[j][k] += delta;
            }
        }
    }
    // Мутация bias выходного слоя
    for k in 0..net.output_size {
        if rng.r#gen::<f32>() < MUTATION_RATE {
            let delta = rng.gen_range(-MUTATION_STRENGTH..MUTATION_STRENGTH);
            net.bias_output[k] += delta;
        }
    }
}


