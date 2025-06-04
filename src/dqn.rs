use crate::network::Network;
use rand::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use crate::snake::*;
use crate::food::*;

const DISCOUNT: f32 = 0.99;
const EPSILON_DECAY: f32 = 0.995;
const MIN_EPSILON: f32 = 0.05;

pub struct DQNAgent {
    pub network: Network,
    pub epsilon: f32,
    pub memory: Vec<(Vec<f32>, usize, f32, Vec <f32>)>, // опыт s, a, r, s
    pub rng: ThreadRng,
}

impl DQNAgent {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: f32) -> Self {
        DQNAgent {
            network: Network::new(input, hidden, output, learning_rate),
            epsilon: 1.0,
            memory: Vec::new(),
            rng: rand::thread_rng(),
        }
    }

    pub fn new_with_network(network: Network) -> Self {
        Self {
            epsilon: 1.0,                     // стартовое значение ε (для ε-greedy стратегии)
            memory: Vec::new(),              // очищенная память (будет заполняться в процессе)
            rng: rand::thread_rng(),         // создаём генератор случайных чисел
            network,                         // <-- вот главное: сюда передаём уже загруженную сеть!
        }
    }

    ///Выбирает дейтсвие случайное или по нейронке
    pub fn select_action(&mut self, state: &[f32]) -> usize {
        if self.rng.r#gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..self.network.output_size)
        } else {
            let q_values = self.network.forward(state);
            q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    //Сохраняем опыт для тренировки
    pub fn store_experience(&mut self, state: Vec<f32>, action: usize, reward: f32, next_state: Vec<f32>) {
        self.memory.push((state, action, reward, next_state));
    }

    //тренируем нейронку на опыте
    pub fn train(&mut self) {
        for (state, action, reward, next_state) in &self.memory {
            let mut target = self.network.forward(state);
            let next_q = self.network.forward(next_state);
            let max_next = next_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            target[*action] = reward + DISCOUNT * max_next;
            self.network.train(state, &target);
        }

        //обновляем epsilon
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(MIN_EPSILON);
        self.memory.clear();
    }

    //записб результатов обучнеия
    pub fn log_episode(&self, episode: usize, score: u32) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("training_log.csv")
            .unwrap();
        writeln!(file, "{episode},{score},{:.3}", self.epsilon).unwrap();
    }
}

pub fn get_state(snake: &Snake, food: &Food) -> Vec<f32> {
    let head = snake.body[0];

    let point_left  = (head.0.wrapping_sub(1), head.1);
    let point_right = (head.0 + 1, head.1);
    let point_up    = (head.0, head.1.wrapping_sub(1));
    let point_down  = (head.0, head.1 + 1);

    // Функция проверяет: выходит ли координата за пределы или врезаемся в себя
    let is_danger = |p: (u32, u32)| -> bool {
        p.0 >= crate::snake::BOARD_WIDTH || p.1 >= crate::snake::BOARD_HEIGHT || snake.body.contains(&p)
    };

    let danger_straight = match snake.direction {
        Direction::Up    => is_danger(point_up),
        Direction::Down  => is_danger(point_down),
        Direction::Left  => is_danger(point_left),
        Direction::Right => is_danger(point_right),
    };

    let danger_right = match snake.direction {
        Direction::Up    => is_danger(point_right),
        Direction::Down  => is_danger(point_left),
        Direction::Left  => is_danger(point_up),
        Direction::Right => is_danger(point_down),
    };

    let danger_left = match snake.direction {
        Direction::Up    => is_danger(point_left),
        Direction::Down  => is_danger(point_right),
        Direction::Left  => is_danger(point_down),
        Direction::Right => is_danger(point_up),
    };

    let dir_up    = snake.direction == Direction::Up;
    let dir_down  = snake.direction == Direction::Down;
    let dir_left  = snake.direction == Direction::Left;
    let dir_right = snake.direction == Direction::Right;

    let food_left  = food.position.0 < head.0;
    let food_right = food.position.0 > head.0;
    let food_up    = food.position.1 < head.1;
    let food_down  = food.position.1 > head.1;

    vec![
        danger_straight as u8 as f32,
        danger_right as u8 as f32,
        danger_left as u8 as f32,
        dir_left as u8 as f32,
        dir_right as u8 as f32,
        dir_up as u8 as f32,
        dir_down as u8 as f32,
        food_left as u8 as f32,
        food_right as u8 as f32,
        food_up as u8 as f32,
        food_down as u8 as f32,
    ]
}