///Здесь определена структура Point для координат на игровом поле и функция генерации случайной точки для размещения еды:
use rand::Rng;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    //returns true if this point is present in the points list
    pub fn is_in(&self, points: &Vec<Point>) -> bool {
        points.iter().any(|p| p.x == self.x && p.y == self.y)
    }
}

// Генерирует случайную точку в пределах [0, width) x [0, height)
pub fn random_point(width: i32, height: i32) -> Point {
    let mut rng = rand::thread_rng();
    let x = rng.r#gen_range(0..width);
    let y = rng.r#gen_range(0..height);
    Point {x , y}
}

impl Point {
    pub fn manhattan(&self, other: Point) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}