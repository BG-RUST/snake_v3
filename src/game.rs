// Game rules module: state, steps, collisions, food, spawn.

use crate::food::*;
use crate::snake::*;
use crate::utils::*;

/// Result of an RL step: immediate reward and termination flag.
pub struct StepOutcome {
    pub reward: f32,
    pub done: bool,
}

pub struct Game {
    // Grid size (in cells).
    w: usize,
    h: usize,

    // Snake state.
    snake: Snake,

    // Food position.
    food: Food,

    // Buffered direction from keyboard (used by step()).
    pending_dir: Dir,

    // Terminal flag for manual mode (UI loop).
    done: bool,

    // RNG for food spawning.
    rng: LcgRng,

    // ---- Extra fields for RL shaping ----
    steps_since_food: u32, // how many steps since last apple
    hunger_limit: u32,     // max steps without food before terminating the episode
    last_manhattan: i32,   // previous Manhattan distance to food (for shaping)
}

impl Game {
    /// Create a new game with centered snake and one food.
    pub fn new(w: usize, h: usize) -> Self {
        let mut rng = LcgRng::new(0xC0FFEE_u64);
        let snake = Snake::new((w / 2) as i32, (h / 2) as i32);
        let mut g = Self {
            w,
            h,
            snake,
            food: Food::at(0, 0),
            pending_dir: Dir::Right,
            done: false,
            rng,
            steps_since_food: 0,
            hunger_limit: 200,
            last_manhattan: 0,
        };
        // Spawn initial food and compute initial Manhattan distance.
        g.respawn_food();
        let (hx, hy) = g.snake.head();
        g.last_manhattan = (g.food.x as i32 - hx).abs() + (g.food.y as i32 - hy).abs();
        g
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.snake = Snake::new((self.w / 2) as i32, (self.h / 2) as i32);
        self.pending_dir = Dir::Right;
        self.done = false;
        self.steps_since_food = 0;
        self.respawn_food();
        let (hx, hy) = self.snake.head();
        self.last_manhattan = (self.food.x as i32 - hx).abs() + (self.food.y as i32 - hy).abs();
    }

    /// One logical step for manual mode (uses `pending_dir`).
    pub fn step(&mut self) {
        if self.done {
            return;
        }

        // Apply desired direction, disallowing instant 180° turns.
        self.snake.apply_dir(self.pending_dir);

        // Move the snake forward by one cell.
        self.snake.advance();

        // Check wall collision.
        let (hx, hy) = self.snake.head();
        if hx < 0 || hy < 0 || hx >= self.w as i32 || hy >= self.h as i32 {
            self.done = true;
            return;
        }

        // Check self-collision.
        if self.snake.self_collision() {
            self.done = true;
            return;
        }

        // Check food.
        if (hx as usize, hy as usize) == (self.food.x, self.food.y) {
            self.snake.feed();   // grow on the next move
            self.respawn_food(); // place a new food
        }
    }

    // ---------------- RL mode ----------------

    /// Step the environment with a relative action in the snake's local frame:
    /// action_rel = 0: turn left, 1: go straight, 2: turn right.
    pub fn step_ai(&mut self, action_rel: u8) -> StepOutcome {
        // Small step penalty to encourage shorter paths.
        let mut reward = -0.01f32;

        // Current heading before the move.
        let cur = self.snake.dir();

        // Map relative action to absolute direction.
        let abs_dir = match (cur, action_rel) {
            (Dir::Up,    0) => Dir::Left,   (Dir::Up,    1) => Dir::Up,    (Dir::Up,    2) => Dir::Right,
            (Dir::Down,  0) => Dir::Right,  (Dir::Down,  1) => Dir::Down,  (Dir::Down,  2) => Dir::Left,
            (Dir::Left,  0) => Dir::Down,   (Dir::Left,  1) => Dir::Left,  (Dir::Left,  2) => Dir::Up,
            (Dir::Right, 0) => Dir::Up,     (Dir::Right, 1) => Dir::Right, (Dir::Right, 2) => Dir::Down,
            _ => cur, // fallback (should not happen)
        };

        // Apply chosen direction (instant reversal still forbidden inside).
        self.snake.apply_dir(abs_dir);

        // Advance one cell.
        self.snake.advance();

        // Wall collision ends the episode with negative reward.
        let (hx, hy) = self.snake.head();
        if hx < 0 || hy < 0 || hx >= self.w as i32 || hy >= self.h as i32 {
            return StepOutcome { reward: -1.0, done: true };
        }

        // Self-collision ends the episode with negative reward.
        if self.snake.self_collision() {
            return StepOutcome { reward: -1.0, done: true };
        }

        // Shaping: progress towards food by Manhattan distance.
        let manh = (self.food.x as i32 - hx).abs() + (self.food.y as i32 - hy).abs();
        let delta = (self.last_manhattan - manh) as f32; // decrease => positive
        let alpha = 0.01f32;                             // small shaping weight
        let delta_clamped = if delta > 1.0 { 1.0 } else if delta < -1.0 { -1.0 } else { delta };
        reward += alpha * delta_clamped;
        self.last_manhattan = manh;

        // Check eating.
        if (hx as usize, hy as usize) == (self.food.x, self.food.y) {
            self.snake.feed();
            self.respawn_food();
            self.steps_since_food = 0;
            reward += 1.0; // big positive reward for food
            return StepOutcome { reward, done: false };
        }

        // Hunger increases if no food eaten.
        self.steps_since_food += 1;

        // Episode ends if too long without food.
        if self.steps_since_food >= self.hunger_limit {
            reward -= 0.2;
            return StepOutcome { reward, done: true };
        }

        // Continue episode.
        StepOutcome { reward, done: false }
    }

    // ---------- Observation for DQN ----------

    /// Dimension of the observation vector.
    /// 5 rays × (wall/body/food) + cos/sin to food + [length, hunger]
    pub fn observation_dim(&self) -> usize { 5 * 3 + 2 + 2 }

    /// Build observation:
    /// - 5 local rays (left, left-forward, forward, right-forward, right),
    ///   for each: normalized distances to wall/body/food;
    /// - unit vector to food in the head's local frame (cos/sin);
    /// - normalized snake length and hunger.
    pub fn observe(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(self.observation_dim());

        // Head position and facing direction.
        let (hx, hy) = self.snake.head();
        let dir = self.snake.dir();

        // Local rays in the head's frame (forward = +y in local coords).
        let rays_local: [(i32, i32); 5] = [
            (-1, 0), // left
            (-1, 1), // left-forward
            ( 0, 1), // forward
            ( 1, 1), // right-forward
            ( 1, 0), // right
        ];

        // Rotate local (dx,dy) into global based on current facing.
        let rot = |dx: i32, dy: i32, d: Dir| -> (i32, i32) {
            match d {
                // Up means global +y is negative (screen coords), so flip dy.
                Dir::Up    => (dx, -dy),
                // Down flips x.
                Dir::Down  => (-dx,  dy),
                // Left swaps and flips both.
                Dir::Left  => (-dy, -dx),
                // Right swaps into (dy,dx).
                Dir::Right => ( dy,  dx),
            }
        };

        // Upper bound for distances used for normalization.
        let max_r = (self.w.max(self.h)) as f32;

        // For each ray, compute 3 distances: to wall, to body, to food.
        for (lx, ly) in rays_local {
            let (dx, dy) = rot(lx, ly, dir);

            let mut dist_wall: f32 = 0.0; // exact wall distance (cells)
            let mut dist_body: f32 = max_r;
            let mut dist_food: f32 = max_r;

            // Scan forward along the ray starting one cell ahead.
            let mut step: usize = 0;
            let mut x = hx;
            let mut y = hy;
            loop {
                step += 1;
                x += dx;
                y += dy;

                // Wall: first out-of-bounds cell.
                if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
                    dist_wall = step as f32;
                    break;
                }
                // Body: first time we see a segment.
                if self.snake.occupies(x, y) && dist_body == max_r {
                    dist_body = step as f32;
                }
                // Food: first time we see the food on this ray.
                if (x as usize, y as usize) == (self.food.x, self.food.y) && dist_food == max_r {
                    dist_food = step as f32;
                }

                // Safety guard to avoid infinite loops.
                if step > (self.w + self.h) {
                    break;
                }
            }

            // If body/food not seen on this ray, treat as "far" (use wall distance).
            if dist_body == max_r { dist_body = dist_wall.max(1.0); }
            if dist_food == max_r { dist_food = dist_wall.max(1.0); }

            // Normalize into [0,1] by wall distance (closer => smaller fraction).
            let norm = dist_wall.max(1.0);
            obs.push((dist_wall / norm).min(1.0)); // wall
            obs.push((dist_body / norm).min(1.0)); // body
            obs.push((dist_food / norm).min(1.0)); // food
        }

        // Unit vector to food in the head's local frame -> (cos, sin).
        let vx = self.food.x as i32 - hx;
        let vy = self.food.y as i32 - hy;
        let (vx_l, vy_l) = rot(vx, vy, dir);
        let len = (((vx_l * vx_l + vy_l * vy_l) as f32).sqrt()).max(1e-6);
        let cos_t = (vy_l as f32) / len; // forward component
        let sin_t = (vx_l as f32) / len; // left/right component
        obs.push(cos_t);
        obs.push(sin_t);

        // Normalized snake length and hunger.
        let len_snake = self.snake.len() as f32 / (self.w * self.h) as f32;
        let hunger = self.steps_since_food as f32 / self.hunger_limit as f32;
        obs.push(len_snake);
        obs.push(hunger);

        obs
    }

    /// External input for manual mode: set desired direction.
    pub fn set_pending_dir(&mut self, d: Dir) { self.pending_dir = d; }

    /// Pick a new food cell uniformly among empty cells.
    fn respawn_food(&mut self) {
        loop {
            let x = self.rng.gen_range_u32(self.w as u32) as usize;
            let y = self.rng.gen_range_u32(self.h as u32) as usize;
            if !self.snake.occupies(x as i32, y as i32) {
                self.food = Food::at(x, y);
                break;
            }
        }
    }

    // -------- getters for rendering / control --------
    pub fn width(&self) -> usize { self.w }
    pub fn height(&self) -> usize { self.h }
    pub fn is_done(&self) -> bool { self.done }
    pub fn food_pos(&self) -> (usize, usize) { (self.food.x, self.food.y) }
    pub fn snake_segments(&self) -> Vec<(i32, i32)> { self.snake.segments_vec() }
}
