use rand::Rng;
use std::fs;
use std::io::{Write, BufRead, BufReader};
use crate::snake::Direction;
use crate::snake::Snake;
use crate::food::Food;

/// Константы архитектуры нейросети
pub const INPUT_COUNT: usize = 121 + 2 + 4;    // число входов нейросети
pub const HIDDEN_COUNT: usize = 10;  // число нейронов в скрытом слое
pub const OUTPUT_COUNT: usize = 4;   // число выходов (4 направления движения)

/// Структура, описывающая нейронную сеть (2 слоя: скрытый и выходной)
#[derive(Clone)]  // реализуем Clone, чтобы можно было копировать сеть (для GA)
pub struct NeuralNet {
    pub input_size: usize,   // число входных нейронов
    pub hidden_size: usize,  // число нейронов скрытого слоя
    pub output_size: usize,  // число выходных нейронов
    // Веса между входным и скрытым слоем (матрица размером input_size x hidden_size)
    pub weights_input_hidden: Vec<Vec<f32>>,
    // Смещения (bias) скрытого слоя (длина hidden_size)
    pub bias_hidden: Vec<f32>,
    // Веса между скрытым и выходным слоем (матрица hidden_size x output_size)
    pub weights_hidden_output: Vec<Vec<f32>>,
    // Смещения выходного слоя (длина output_size)
    pub bias_output: Vec<f32>,
}

impl NeuralNet {
    /// Создает новую нейронную сеть с указанным количеством нейронов в слоях.
    /// Инициализирует все веса и смещения случайными значениями.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
      //  let input_size = 11;
        // Инициализируем веса небольшими случайными значениями (например, в диапазоне -1.0..1.0)
        let weights_input_hidden = (0..input_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-1.0..1.0))  // случайный вес для каждой связи вход-скрытый
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let bias_hidden = (0..hidden_size)
            .map(|_| rng.gen_range(-1.0..1.0)) // случайные смещения скрытого слоя
            .collect::<Vec<f32>>();

        let weights_hidden_output = (0..hidden_size)
            .map(|_| {
                (0..output_size)
                    .map(|_| rng.gen_range(-1.0..1.0))  // случайный вес для каждой связи скрытый-выход
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();

        let bias_output = (0..output_size)
            .map(|_| rng.gen_range(-1.0..1.0))  // случайные смещения выходного слоя
            .collect::<Vec<f32>>();

        Self {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            bias_hidden,
            weights_hidden_output,
            bias_output,
        }
    }

    /// Прямой проход (forward propagation) через нейронную сеть.
    /// Принимает на вход вектор входных значений (длины input_size).
    /// Возвращает массив из output_size выходных значений.
    pub fn forward(&self, inputs: &[f32]) -> [f32; OUTPUT_COUNT] {
        assert!(inputs.len() == self.input_size, "Размер входного вектора не соответствует ожидаемому");
        // Массив для результатов скрытого слоя
        let mut hidden_outputs = vec![0.0; self.hidden_size];

        // Вычисляем активации скрытого слоя: для каждого нейрона скрытого слоя суммируем взвешенные входы и добавляем смещение, затем применяем функцию активации.
        for j in 0..self.hidden_size {
            // смещение (bias) скрытого нейрона j
            let mut sum = self.bias_hidden[j];
            // добавляем взвешенный вклад от каждого входа i
            for i in 0..self.input_size {
                sum += self.weights_input_hidden[i][j] * inputs[i];
            }
            // Применяем нелинейную функцию активации (tanh) к сумме.
            // tanh (гиперболический тангенс) ограничивает выход в диапазоне [-1, 1].
            // Формула: tanh(sum) = (e^sum - e^{-sum}) / (e^sum + e^{-sum})
            let activated = if sum > 20.0 {
                1.0  // tanh -> 1 при больших положительных sum (избегаем переполнения exp)
            } else if sum < -20.0 {
                -1.0 // tanh -> -1 при больших отрицательных sum
            } else {
                let e_pos = sum.exp();
                let e_neg = (-sum).exp();
                (e_pos - e_neg) / (e_pos + e_neg)
            };
            hidden_outputs[j] = activated;
        }

        // Теперь вычисляем выходы сети (слой выхода).
        let mut outputs = [0.0; OUTPUT_COUNT];
        for k in 0..self.output_size {
            // начинаем с bias выходного нейрона k
            let mut sum = self.bias_output[k];
            // добавляем вклад от каждого нейрона скрытого слоя j
            for j in 0..self.hidden_size {
                sum += self.weights_hidden_output[j][k] * hidden_outputs[j];
            }
            // На выходном слое можно не применять функцию активации,
            // чтобы значения могли свободно варьироваться (мы будем выбирать максимальный).
            outputs[k] = sum;
        }
        outputs
    }

    /// Преобразует выходной массив значений в конкретное направление движения.
    /// (Выбирается направление с максимальным значением на выходе сети)
    fn output_to_direction(&self, output: &[f32; OUTPUT_COUNT]) -> Direction {
        // Находим индекс максимального значения в output
        let mut max_index = 0;
        let mut max_value = output[0];
        for i in 1..output.len() {
            if output[i] > max_value {
                max_value = output[i];
                max_index = i;
            }
        }
        // Соотносим индекс с направлением: 0-Up, 1-Down, 2-Left, 3-Right
        match max_index {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => Direction::Up, // на всякий случай (сюда не зайдем, так как индексы 0-3)
        }
    }

    /// Подготавливает входной вектор для нейросети на основе текущего состояния игры (змейки и еды).
    fn prepare_inputs(snake: &Snake, food: &Food, width: u32, height: u32) -> Vec<f32> {
        let (head_x, head_y) = snake.body[0];
        let cols = (width / 20) as i32;
        let rows = (height / 20) as i32;
        let head_x = head_x as i32;
        let head_y = head_y as i32;

        let mut vision = Vec::with_capacity(INPUT_COUNT);

        let vision_radius = 5; // 11x11 = 121
        for dy in -vision_radius..=vision_radius {
            for dx in -vision_radius..=vision_radius {
                let x = head_x + dx;
                let y = head_y + dy;

                let val = if x < 0 || y < 0 || x >= cols || y >= rows {
                    1.0 // стена
                } else if snake.body.contains(&(x as u32, y as u32)) {
                    0.5 // тело
                } else if food.position == (x as u32, y as u32) {
                    0.9 // еда
                } else {
                    0.0 // пусто
                };
                vision.push(val);
            }
        }

        // Направление к еде
        let dx = (food.position.0 as i32 - head_x) as f32 / cols as f32;
        let dy = (food.position.1 as i32 - head_y) as f32 / rows as f32;
        vision.push(dx);
        vision.push(dy);

        // One-hot направление змейки
        let mut dir_encoding = [0.0; 4];
        match snake.next_dir {
            Direction::Up => dir_encoding[0] = 1.0,
            Direction::Down => dir_encoding[1] = 1.0,
            Direction::Left => dir_encoding[2] = 1.0,
            Direction::Right => dir_encoding[3] = 1.0,
        }
        vision.extend_from_slice(&dir_encoding);

        assert_eq!(vision.len(), INPUT_COUNT, "prepare_inputs() вернул вектор неправильной длины");
        vision
    }



    /// Решение нейросети: вычисляет направление движения змейки на основе текущего состояния.
    /// Использует внутреннюю сеть (forward) и возвращает направление (тип Direction).
    pub fn decide_direction(&self, snake: &Snake, food: &Food, width: u32, height: u32) -> Direction {

       // let inputs = NeuralNet::prepare_inputs(snake, food, width, height);
       // if inputs.iter().any(|v| (*v - 0.9).abs() < 0.01) {
        //    println!("Сеть видит еду! 🧠");
        //}

        // Подготавливаем входы
        let inputs = NeuralNet::prepare_inputs(snake, food, width, height);
        // Прогоняем входы через нейронную сеть, получаем выходной вектор
        let output = self.forward(&inputs);
        // Преобразуем выходной вектор в одно из направлений (выбираем максимальный выход)
        self.output_to_direction(&output)
    }

    /// Сохраняет веса нейронной сети в файл (для продолжения обучения в будущем).
    pub fn save_to_file(&self, path: &str) {
        let mut file = fs::File::create(path).expect("Не удалось создать файл для сохранения весов");
        // Записываем размеры сети (input, hidden, output) первой строкой
        writeln!(file, "{} {} {}", self.input_size, self.hidden_size, self.output_size).unwrap();
        // Записываем все веса input->hidden
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                write!(file, "{} ", self.weights_input_hidden[i][j]).unwrap();
            }
        }
        writeln!(file).unwrap();
        // Записываем bias скрытого слоя
        for j in 0..self.hidden_size {
            write!(file, "{} ", self.bias_hidden[j]).unwrap();
        }
        writeln!(file).unwrap();
        // Записываем все веса hidden->output
        for j in 0..self.hidden_size {
            for k in 0..self.output_size {
                write!(file, "{} ", self.weights_hidden_output[j][k]).unwrap();
            }
        }
        writeln!(file).unwrap();
        // Записываем bias выходного слоя
        for k in 0..self.output_size {
            write!(file, "{} ", self.bias_output[k]).unwrap();
        }
        writeln!(file).unwrap();
        // В конце файла каждая часть записывается с пробелами, по строкам для удобства.
    }

    /// Загружает веса нейронной сети из файла. Если не удалось загрузить, возвращает None.
    pub fn load_from_file(path: &str) -> Option<Self> {
        let file = fs::File::open(path).ok()?; // открываем файл, если не удалось - возвращаем None
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Читаем первую строку с размерами сети
        let line1 = lines.next()?.ok()?;
        let dims: Vec<usize> = line1
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        if dims.len() != 3 {
            println!("Формат файла весов некорректен (не содержит три числа размеров).");
            return None;
        }
        let (input_size, hidden_size, output_size) = (dims[0], dims[1], dims[2]);

        // Создаем пустую сеть нужного размера (с случайными весами, мы их перезапишем)
        let mut net = NeuralNet::new(input_size, hidden_size, output_size);

        // Читаем следующую строку - веса input->hidden (input_size * hidden_size чисел)
        let line2 = lines.next()?.ok()?;
        let vals: Vec<f32> = line2
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        if vals.len() != input_size * hidden_size {
            println!("Количество весов input-hidden в файле не соответствует ожидаемому.");
            return None;
        }
        // Заполняем веса input->hidden из списка значений
        let mut idx = 0;
        for i in 0..input_size {
            for j in 0..hidden_size {
                net.weights_input_hidden[i][j] = vals[idx];
                idx += 1;
            }
        }

        // Читаем следующую строку - bias_hidden
        let line3 = lines.next()?.ok()?;
        let vals: Vec<f32> = line3
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        if vals.len() != hidden_size {
            println!("Количество bias скрытого слоя в файле не соответствует ожидаемому.");
            return None;
        }
        for j in 0..hidden_size {
            net.bias_hidden[j] = vals[j];
        }

        // Читаем следующую строку - веса hidden->output
        let line4 = lines.next()?.ok()?;
        let vals: Vec<f32> = line4
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        if vals.len() != hidden_size * output_size {
            println!("Количество весов hidden-output в файле не соответствует ожидаемому.");
            return None;
        }
        idx = 0;
        for j in 0..hidden_size {
            for k in 0..output_size {
                net.weights_hidden_output[j][k] = vals[idx];
                idx += 1;
            }
        }

        // Читаем последнюю строку - bias_output
        let line5 = lines.next()?.ok()?;
        let vals: Vec<f32> = line5
            .split_whitespace()
            .filter_map(|w| w.parse().ok())
            .collect();
        if vals.len() != output_size {
            println!("Количество bias выходного слоя в файле не соответствует ожидаемому.");
            return None;
        }
        for k in 0..output_size {
            net.bias_output[k] = vals[k];
        }

        Some(net)
    }
}
