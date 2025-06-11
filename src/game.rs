use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use pixels::{Pixels, SurfaceTexture};

use crate::snake::Snake;
use crate::food::Food;
use crate::border::Border;
use crate::NeuralNet;

use std::time::{Instant, Duration};
use winit::platform::run_return::EventLoopExtRunReturn;

pub struct Game {
    width: usize,
    height: usize,
    snake: Snake,
    food: Food,
    border: Border,
}

impl Game {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            snake: Snake::new(),
            food: Food::new(),
            border: Border::new((width / 32) as u32, (height / 32) as u32),
        }
    }

    pub fn run_with_ai(mut self, net: &NeuralNet, event_loop: &mut EventLoop<()>) -> u32 {
        let window = WindowBuilder::new()
            .with_title("Snake Ai")
            .with_inner_size(winit::dpi::LogicalSize::new(self.width as f64, self.height as f64))
            .build(&event_loop)
            .unwrap();

        let surface_texture = SurfaceTexture::new(self.width as u32, self.height as u32, &window);
        let mut pixels = Pixels::new(self.width as u32, self.height as u32, surface_texture).unwrap();

        let mut last_update = Instant::now();
        let mut score = 0;
        let mut steps_without_food = 0;
        const MAX_STEPS_WITHOUT_FOOD: u32 = 200;

        event_loop.run_return(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::MainEventsCleared => {
                    if last_update.elapsed() > Duration::from_millis(1) {
                        // inputs для нейросети
                        let inputs = Self::get_inputs(&self);

                        let action = net.forward(&inputs);
                        Self::apply_action(&mut self.snake, action);

                        self.snake.update();
                        steps_without_food += 1;

                        let (x, y) = self.snake.head();
                        if x <= 0 || y <= 0 || x as u32 >= self.border.width - 1 || y as u32 >= self.border.height - 1
                            || self.snake.is_colliding_with_self()
                            || steps_without_food >= MAX_STEPS_WITHOUT_FOOD
                        {
                            println!(
                                "☠ Умерла | score: {} | steps: {}",
                                score, steps_without_food
                            );
                            *control_flow = ControlFlow::Exit;
                            return;
                        }

                        if self.snake.head() == self.food.position {
                            self.snake.grow();
                            self.food.respawn(&self.snake.body, self.border.width, self.border.height);
                            score += 1;
                            steps_without_food = 0; // сброс счётчика
                        }

                        window.request_redraw();
                        last_update = Instant::now();
                    }
                }

                Event::RedrawRequested(_) => {
                    let frame = pixels.frame_mut();
                    frame.fill(0);

                    self.draw_border(frame);
                    self.food.draw(frame, self.width);
                    self.snake.draw(frame);

                    if pixels.render().is_err() {
                        *control_flow = ControlFlow::Exit;
                    }
                }

                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                }

                _ => {}
            }
        });

        score
    }


    fn get_inputs(game: &Game) -> [f32; 6] {
        let (hx, hy) = game.snake.head();
        let dir = game.snake.direction();

        let left = (-dir.1, dir.0);
        let right = (dir.1, -dir.0);
        let forward = dir;

        fn look((x, y): (i32, i32), dir: (i32, i32)) -> (i32, i32) {
            (x + dir.0, y + dir.1)
        }
        let mut inputs = [0.0; 6];

        for (i, d) in [forward, left, right].iter().enumerate() {
            let p = look((hx, hy), *d);
            let wall = p.0 <= 0 || p.1 <= 0 || p.0 as u32 >= game.border.width - 1 || p.1 as u32 >= game.border.height - 1;
            let food = p == game.food.position;
            let tail = game.snake.body.contains(&p);

            inputs[i * 2] = if wall || tail {1.0} else {0.0};
            inputs[i * 2 + 1] = if food {1.0} else {0.0};
        }
        inputs
    }

    fn apply_action(snake: &mut Snake, action: usize) {
        let (dx, dy) = snake.direction();
        let new_dir = match action {
            0 => (-dy, dx),//left
            1 => (dx, dy), //forward
            2 => (dy, -dx), //right
            _ => (dx, dy),

        };
        snake.set_dir(new_dir);
    }

    pub fn run(mut self, event_loop: &mut EventLoop<()>) {
        //let mut event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title("Snake with winit")
            .with_inner_size(winit::dpi::LogicalSize::new(self.width as f64, self.height as f64))
            .build(&event_loop)
            .unwrap();

        let surface_texture = SurfaceTexture::new(self.width as u32, self.height as u32, &window);
        let mut pixels = Pixels::new(self.width as u32, self.height as u32, surface_texture).unwrap();

        let mut last_update = Instant::now();

        event_loop.run_return(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,

                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(key),
                            ..
                        },
                        ..
                    },
                    ..
                } => {
                    match key {
                        VirtualKeyCode::W => self.snake.set_dir((0, -1)),
                        VirtualKeyCode::S => self.snake.set_dir((0, 1)),
                        VirtualKeyCode::A => self.snake.set_dir((-1, 0)),
                        VirtualKeyCode::D => self.snake.set_dir((1, 0)),
                        _ => {}
                    }
                }

                Event::MainEventsCleared => {
                    if last_update.elapsed() >= Duration::from_millis(150) {
                        self.snake.update();

                        // столкновение с границей — завершить игру
                        let (x, y) = self.snake.head();
                        if x <= 0 || y <= 0 || x as u32 >= self.border.width - 1 || y as u32 >= self.border.height - 1 {
                            println!("Game over! 🧱");
                            *control_flow = ControlFlow::Exit;
                            return;
                        }

                        if self.snake.is_colliding_with_self() {
                            println!("Game over!");
                            *control_flow = ControlFlow::Exit;
                            return;
                        }

                        if self.snake.head() == self.food.position {
                            self.snake.grow();
                            self.food.respawn(&self.snake.body, self.border.width, self.border.height);
                        }

                        window.request_redraw();
                        last_update = Instant::now();
                    }
                }

                Event::RedrawRequested(_) => {
                    let frame = pixels.frame_mut();
                    frame.fill(0); // очистка экрана

                    self.draw_border(frame);
                    self.food.draw(frame, self.width);
                    self.snake.draw(frame);

                    if pixels.render().is_err() {
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                }

                _ => {}
            }
        });
    }

    fn draw_border(&self, frame: &mut [u8]) {
        let cell_size = 32;
        for x in 0..self.border.width {
            for y in 0..self.border.height {
                if x == 0 || y == 0 || x == self.border.width - 1 || y == self.border.height - 1 {
                    for dy in 0..cell_size {
                        for dx in 0..cell_size {
                            let px = (x * cell_size + dx as u32) as usize;
                            let py = (y * cell_size + dy as u32) as usize;
                            let i = (py * self.width + px) * 4;
                            if i + 4 <= frame.len() {
                                frame[i..i + 4].copy_from_slice(&[128, 128, 128, 255]); // серые границы
                            }
                        }
                    }
                }
            }
        }
    }
}
