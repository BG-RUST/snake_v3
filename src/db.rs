// db.rs

use std::fs::File;
use std::io::{BufReader, Write, Error};
use serde::{Serialize, Deserialize};
use crate::network::Network;
use serde_json;

// Структура для сохранения состояния обучения
#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub network: Network,
    pub epsilon: f32,
}

// Сохраняет текущую модель и состояние обучения в файл (JSON формат)
pub fn save(save_data: &SaveData, path: &str) -> Result<(), Error> {
    let json = serde_json::to_string(save_data).unwrap();
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

// Загружает модель и состояние обучения из файла
pub fn load(path: &str) -> Result<SaveData, Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let save_data: SaveData = serde_json::from_reader(reader).unwrap();
    Ok(save_data)
}
