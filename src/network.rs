use rand::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write, BufRead};


fn relu(x: f32) -> f32 {
    x.max(0.0)
}
// Производная ReLU для обратного распространения ошибки
fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {1.0} else {0.0}
}

pub struct Network {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub weights_ih: Vec<Vec<f32>>, //input -> hidden
    pub weights_ho: Vec<Vec<f32>>, //hidden output
    pub learning_rate: f32,
}

impl Network {
    pub fn new(input: usize, hidden: usize, output: usize, lr: f32) -> Self {
        let mut rng = rand::thread_rng();

        let weights_ih = (0..hidden)
            .map(|_| (0..input).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let weights_ho = (0..output)
            .map(|_| (0..hidden).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        Self {
            input_size: input,
            hidden_size: hidden,
            output_size: output,
            weights_ih,
            weights_ho,
            learning_rate: lr,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let hidden: Vec<f32> = self.weights_ih.iter()
            .map(|weights| relu(weights.iter().zip(input).map(|(w, i)| w * i).sum()))
            .collect();

        self.weights_ho.iter()
            .map(|weights| weights.iter().zip(&hidden).map(|(w, h)| w * h).sum())
            .collect()
    }

    // Обратное распространение ошибки (однослойное)
    pub fn train(&mut self, input: &[f32], target: &[f32]) {
        //прямой прозод

        let hidden: Vec<f32> = self.weights_ih.iter()
            .map(|weights| relu(weights.iter().zip(input).map(|(w, i)| w * i).sum()))
            .collect();

        let output: Vec<f32> = self.weights_ho.iter()
            .map(|weights| weights.iter().zip(&hidden).map(|(w, h)| w * h).sum())
            .collect();

        // ошибка на выходе + обновление весов hidden -> output
        for i in 0..self.output_size {
            let error = target[i] - output[i];
            for j in 0..self.hidden_size {
                let delta = error * hidden[j];
                self.weights_ho[i][j] += self.learning_rate * delta;
            }
        }

        // ошибка на скрытом слое + обновление весов input -> hidden
        for j in 0..self.hidden_size {
            let hidden_output = hidden[j];
            let grad = relu_derivative(hidden_output);
            let sum: f32 = (0..self.output_size)
                .map(|i| (target[i] - output[i]) * self.weights_ho[i][j])
                .sum();

            for k in 0..self.input_size {
                let delta = grad * sum * input[k];
                self.weights_ih[j][k] += self.learning_rate * delta;
            }
        }
    }

    //сохраняем
    pub fn save_weights_csv(&self, path: &str) -> std::io::Result<()> {

        let mut file = BufWriter::new(File::create(path)?);

        writeln!(file, "{},{},{},{}", self.input_size, self.hidden_size, self.output_size, self.learning_rate)?;

        for row in &self.weights_ih {
            let line = row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            writeln!(file, "ih, {}", line)?;
        }

        for row in &self.weights_ho {
            let line = row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
            writeln!(file, "ho, {}", line)?;
        }

        Ok(())
    }

    //загружаем модель из текстового файла
    pub fn load_weights_csv(path: &str) -> std::io::Result<Self> {
        let file = BufReader::new(File::open(path)?);
        let mut lines = file.lines();

        let header = lines.next().unwrap()?;
        let mut parts = header.split(',');
        let input_size = parts.next().unwrap().parse().unwrap_or(4);
        let hidden_size = parts.next().unwrap().parse().unwrap_or(64);
        let output_size = parts.next().unwrap().parse().unwrap_or(4);
        let learning_rate = parts.next().unwrap().parse().unwrap_or(0.01);

        let mut weights_ih = Vec::new();
        let mut weights_ho = Vec::new();

        for line in lines {
            let line = line?;
            let mut parts = line.split(',');
            let tag = parts.next().unwrap_or("").trim();

            let row: Vec<f32> = parts
                .map(|s| s.trim().parse::<f32>())
                .filter_map(Result::ok)
                .collect();

            if row.is_empty() {
                eprintln!("⚠ Пропущена строка с некорректными значениями: {}", line);
                continue;
            }

            match tag {
                "ih" => weights_ih.push(row),
                "ho" => weights_ho.push(row),
                _ => eprintln!("⚠ Неизвестный тег в строке: {}", tag),
            }
        }

        Ok(Self {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            learning_rate,
        })
    }
}