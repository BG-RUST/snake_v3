use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode, ElementState},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use pixels::{Pixels, SurfaceTexture};

use crate::snake::Snake;
use crate::food::Food;
use crate::border::Border;

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
                            self.food.respawn(&self.snake.body);
                        }

                        window.request_redraw();
                        last_update = Instant::now();
                    }
                }

                Event::RedrawRequested(_) => {
                    let frame = pixels.frame_mut();
                    frame.fill(0); // очистка экрана

                    self.draw_border(frame);
                    self.food.draw(frame);
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
