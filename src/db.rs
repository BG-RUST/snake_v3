use std::fs::File;
use std::io::{BufReader, Write, Error};
use serde::{Serialize, Deserialize};
use crate::network::*;
use serde_json;

#[derive(Serialize, Deserialize)]
pub struct SaveData {
    pub network: SerializableNetwork,
    pub epsilon: f32,
}

pub fn save(save_data: &SaveData, path: &str) -> Result<(), Error> {
    let json = serde_json::to_string_pretty(save_data)?; // читаемый JSON
    let mut file = File::create(path)?;
    file.write_all(json.as_bytes())?;
    Ok(())
}

pub fn load(path: &str) -> Result<SaveData, Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let save_data = serde_json::from_reader(reader)?;
    Ok(save_data)
}
