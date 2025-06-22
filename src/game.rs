use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use pixels::{Pixels, SurfaceTexture};
use std::sync::{Arc, Mutex};

use crate::snake::*;
use crate::food::*;
use crate::net::NeuralNet;

pub fn run_game_loop<T>(window: Window, event_loop: EventLoop<T>, best_net: Arc<Mutex<NeuralNet>>)
where
    T: 'static,
{
    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let cell_size = 20;

    let mut pixels = Pixels::new(window_size.width, window_size.height, surface_texture)
        .expect("Не удалось создать Pixel buffer");

    let mut snake = Snake::new();
    let mut food = Food::new_safe(window_size.width, window_size.height, cell_size, &snake);

    let mut last_update = std::time::Instant::now();
    let tick_rate = std::time::Duration::from_millis(150);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("Окно закрывается...");
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },

            Event::MainEventsCleared => {
                if last_update.elapsed() > tick_rate {
                    let direction = {
                        let net = best_net.lock().unwrap();
                        net.decide_direction(&snake, &food, window_size.width, window_size.height)
                    };
                    snake.change_dir(direction);
                    snake.update();

                    let head = snake.body[0];
                    let cols = window_size.width / cell_size;
                    let rows = window_size.height / cell_size;
                    let hit_wall = head.0 <= 1 || head.1 <= 1 || head.0 >= cols - 1 || head.1 >= rows - 1;
                    let hit_self = snake.is_colliding_with_self();

                    if hit_wall || hit_self {
                        println!("Game Over! (змейка погибла)");
                        snake = Snake::new();
                        food = Food::new_safe(window_size.width, window_size.height, cell_size, &snake);
                        last_update = std::time::Instant::now();
                        window.request_redraw();
                        return;
                    }

                    if head == food.position {
                        snake.grow();
                        food = Food::new_safe(window_size.width, window_size.height, cell_size, &snake);
                    }

                    last_update = std::time::Instant::now();
                }
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let frame = pixels.frame_mut();
                fill_background(frame, [30, 30, 30, 255]);
                draw_borders(frame, window_size.width, window_size.height, cell_size, [255, 120, 120, 233]);
                snake.draw(frame, cell_size, window_size.width);
                food.draw(frame, cell_size, window_size.width);
                pixels.render().expect("Ошибка при отрисовке frame");
            }

            _ => {}
        }
    });
}

fn fill_background(frame: &mut [u8], color: [u8; 4]) {
    for chunk in frame.chunks_exact_mut(4) {
        chunk.copy_from_slice(&color);
    }
}

fn draw_borders(frame: &mut [u8], width: u32, height: u32, cell_size: u32, color: [u8; 4]) {
    let screen_width = width as usize;
    let screen_height = height as usize;
    let cell = cell_size as usize;

    for y in 0..screen_height / cell {
        for x in 0..screen_width / cell {
            let is_border = x == 0 || y == 0 || x == (screen_width / cell - 1) || y == (screen_height / cell - 1);

            if is_border {
                for py in 0..cell {
                    for px in 0..cell {
                        let fx = x * cell + px;
                        let fy = y * cell + py;
                        let index = (fy * screen_width + fx) * 4;

                        if index + 3 < frame.len() {
                            frame[index..index + 4].copy_from_slice(&color);
                        }
                    }
                }
            }
        }
    }
}

pub fn draw_cell(
    frame: &mut [u8],
    x: u32,
    y: u32,
    cell_size: u32,
    screen_width: u32,
    color: [u8; 4],
) {
    let cell = cell_size as usize;
    let width = screen_width as usize;

    for py in 0..cell {
        for px in 0..cell {
            let fx = (x * cell_size + px as u32) as usize;
            let fy = (y * cell_size + py as u32) as usize;
            let index = (fy * width + fx) * 4;

            if index + 3 < frame.len() {
                frame[index..index + 4].copy_from_slice(&color);
            }
        }
    }
}
