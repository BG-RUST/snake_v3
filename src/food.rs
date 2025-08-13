pub struct Food {
    pub x: usize,
    pub y: usize,
}

impl Food {
    pub fn at(x: usize, y: usize) -> Self {Self {x, y}}
}