#[derive(Debug)]
pub struct Snake {
    pub body: Vec<(i32, i32)>,
    dir: (i32, i32),
}

impl Snake {
    pub fn new() -> Self {
        Self {
            body: vec![(10, 10)],
            dir: (1, 0),
        }
    }

    pub fn head(&self) -> (i32, i32) {
        self.body[0];
    }

    pub fn update(&mut self) {
        let mut new_head = self.head();
        new_head.0 += self.dir.0;
        new_head.1 += self.dir.1;
        self.body.insert(0, new_head);
        self.body.pop();

    }

    pub fn grow (&mut self) {
        let tail = *self.body.last().unwrap();
        self.body.push(tail);
    }

    pub fn update(&mut self) {

    }

    pub fn grow(&mut self) {

    }
}