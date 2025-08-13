//game rules module (state, steps, collisions, food, spawn).

use crate::food::*;
use crate::snake::*;
use crate::utils::*;

pub struct Game {
    //grid size
    w: usize,
    h: usize,
    //snake
    snake: Snake,
    //food position
    food: Food,
    //direction buffer from input. used during step()
    pending_dir: Dir,
    //episode completion flag
    done: bool,
    //RNG from food spawn
    rng: LcgRng,
}

impl Game {
    //create new game with the snake centrered and one food
    pub fn new(w: usize, h: usize) -> Self {
        let mut rng = LcgRng::new(0xC0FFEE_u64);
        let mut g = Self {
            w,
            h,
            snake: Snake::new((w / 2) as i32, (h / 2) as i32),
            food: Food::at(0, 0),
            pending_dir: Dir::Right,
            done: false,
            rng,
        };
        g.respawn_food();
        g
    }

    //game reset from start state
    pub fn reset(&mut self) {
        self.snake = Snake::new((self.w / 2) as i32, (self.h / 2) as i32);
        self.pending_dir = Dir::Right;
        self.done = false;
        self.respawn_food();
    }

    //one logical step: apply direction, move, check events
    pub fn step(&mut self) {
        if self.done {
            return;
        }

        //change the actual direction, disabling instant 180 turn
        self.snake.apply_dir(self.pending_dir);
        //move the snake 1 cell forward in the current direction
        self.snake.advance();
        //check for collision with wall
        let (hx, hy) = self.snake.head();
        if hx < 0 || hy < 0 || hx >= self.w as i32 || hy >= self.h as i32 {
            self.done = true;
            return;
        }

        //check for collision with itself
        if self.snake.self_collision() {
            self.done = true;
            return;
        }

        //check food
        if (hx as usize, hy as usize) == (self.food.x, self.food.y) {
            //increase the length on next move
            self.snake.feed();
            //respawn food if empty cell
            self.respawn_food();
        }
    }

    //reset the desired direction from outside(keyboard)
    pub fn set_pending_dir(&mut self, d: Dir) {self.pending_dir = d;}

    //select a new food position in a random empty cell
    fn respawn_food(&mut self) {
        //repeat until we find a free cell ( the field is small - ok)
        loop {
            let x = self.rng.gen_range_u32(self.w as u32) as usize;
            let y = self.rng.gen_range_u32(self.h as u32) as usize;
            if !self.snake.occupies(x as i32, y as i32) {
                self.food = Food::at(x, y);
                break;
            }
        }
    }

    //render generators
    pub fn width(&self) -> usize { self.w}
    pub fn height(&self) -> usize { self.h }
    pub fn is_done(&self) -> bool { self.done }
    pub fn food_pos(&self) -> (usize, usize) { (self.food.x, self.food.y) }
    //pub fn snake_segments(&self) -> &[(i32, i32)] {self.snake.segments()}
    pub fn snake_segments(&self) -> Vec<(i32,i32)> {
        self.snake.segments_vec()
    }
}


