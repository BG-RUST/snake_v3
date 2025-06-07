#[derive(Debug)]
pub struct Snake {
    pub body: Vec<(i32, i32)>,
}

impl Snake {
    pub fn new() -> Self {
        Self {
            body: vec![10, 10],
        }
    }

    pub fn update(&mut self) {

    }

    pub fn grow(&mut self) {

    }
}