#[derive(Debug)]
pub struct Food {
    pub position: (i32, i32),
}

impl Food {
    pub fn new() -> Self {
        Self {
            position: (5, 5),
        }
    }

    pub fn respawn(&mut self) {

    }
}