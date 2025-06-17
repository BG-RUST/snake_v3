use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use pixels::{Pixels, SurfaceTexture};
use std::cell::RefCell;
use crate::food::*;
use crate::snake::*;

pub fn run_game_loop<T>(window: Window, event_loop: EventLoop<T>)
where
    T: 'static,
{
    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let cell_size = 20;
    let mut pixels = Pixels::new(
        window_size.width,
        window_size.height,
        surface_texture,
    ).unwrap();

    let mut snake = Snake::new();
    let mut food = Food::new(window_size.width, window_size.height, cell_size);

    let mut last_update = std::time::Instant::now();
    let tick_rate = std::time::Duration::from_millis(150);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
       // let mut pixels = pixels;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("Окно закрывается");
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        use crate::snake::Direction::*;
                        match keycode {
                            winit::event::VirtualKeyCode::W => snake.change_dir(Up),
                            winit::event::VirtualKeyCode::S => snake.change_dir(Down),
                            winit::event::VirtualKeyCode::A => snake.change_dir(Left),
                            winit::event::VirtualKeyCode::D => snake.change_dir(Right),
                            _ => {}
                        }
                    }
                }
                _ => {}
            },



            Event::MainEventsCleared => {
                window.request_redraw();
                if last_update.elapsed() > tick_rate {
                    snake.update();

                    if snake.body[0] == food.position {
                        snake.grow();
                        food = Food::new(window_size.width, window_size.height, cell_size);
                    }
                    last_update = std::time::Instant::now();
                }
            }

            Event::RedrawRequested(_) => {
                let frame = pixels.frame_mut();
                fill_background(frame, [30, 30, 30, 255]); // темно-серый фон
                draw_borders(frame, window_size.width, window_size.height, cell_size, [255, 120, 120, 233]);

                snake.draw(frame, cell_size, window_size.width);
                food.draw(frame, cell_size, window_size.width);

                pixels.render().expect("Ошибка при отрисовке");

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