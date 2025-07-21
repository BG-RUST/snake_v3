use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

#[derive(Serialize, Deserialize, Debug)]
pub struct DqnMetadata {
    pub episode: usize,
    pub epsilon: f32,
    pub average_reward: f32,
}
#[derive(Serialize, Deserialize, Debug)]
pub struct DqnModel {
    pub weights: Vec<Vec<Vec<f32>>>, // <--- 3 слоя × веса нейронов
    pub biases: Vec<Vec<f32>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DqnCheckpoint {
    pub meta: DqnMetadata,
    pub model: DqnModel,
}

///save in json file
pub fn save_checkpoint(path: &str, checkpoint: &DqnCheckpoint) {
    let json = serde_json::to_string(&checkpoint)
        .expect("Could not serialize checkpoint");
    fs::write(path, json)
        .expect("Could not write checkpoint");
}

pub fn load_checkpoint(path: &str) -> Option<DqnCheckpoint> {
    if !Path::new(path).exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()

}

pub fn load_brain() -> Option<DqnModel> {
    load_checkpoint("checkpoint.json").map(|c| c.model)
}