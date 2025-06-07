#[derive(Debug)]
pub struct Border {
    pub width: u32,
    pub height: u32,
}

impl Border {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn is_inside(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }
}