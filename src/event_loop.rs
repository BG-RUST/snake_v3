use winit::event::{Event, WindowEvent, VirtualKeyCode, ElementState};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;
use pixels::{Pixels, SurfaceTexture};
use std::time::{Instant, Duration};
use crate::game::Game;
use crate::snake::Direction;
use crate::network::*;
use crate::db::load;

const CELL_SIZE: u32 = 20;

fn draw_frame(frame: &mut [u8], game: &Game) {
    let width = game.width as u32;
    let height = game.height as u32;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let xi = x as i32;
            let yi = y as i32;

            let (r, g, b) = if game.snake.body[0].x == xi && game.snake.body[0].y == yi {
                (0, 255, 0) // голова
            } else if game.snake.body.iter().skip(1).any(|p| p.x == xi && p.y == yi) {
                (0, 160, 0) // тело
            } else if game.food.position.x == xi && game.food.position.y == yi {
                (255, 0, 0) // еда
            } else {
                (0, 0, 0) // фон
            };

            frame[idx] = r;
            frame[idx + 1] = g;
            frame[idx + 2] = b;
            frame[idx + 3] = 0xFF;
        }
    }
}

pub fn run_manual(mut game: Game) {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake RL - Manual")
        .with_inner_size(winit::dpi::LogicalSize::new(
            (game.width * CELL_SIZE as i32) as f64,
            (game.height * CELL_SIZE as i32) as f64,
        ))
        .build(&event_loop)
        .unwrap();

    let surface_texture = SurfaceTexture::new(window.inner_size().width, window.inner_size().height, &window);
    let mut pixels = Pixels::new(game.width as u32, game.height as u32, surface_texture).unwrap();

    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(100);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => {
                if let WindowEvent::CloseRequested = event {
                    *control_flow = ControlFlow::Exit;
                }

                if let WindowEvent::KeyboardInput { input, .. } = event {
                    if let Some(key) = input.virtual_keycode {
                        if input.state == ElementState::Pressed {
                            game.snake.direction = match key {
                                VirtualKeyCode::Up if game.snake.direction != Direction::Down => Direction::Up,
                                VirtualKeyCode::Down if game.snake.direction != Direction::Up => Direction::Down,
                                VirtualKeyCode::Left if game.snake.direction != Direction::Right => Direction::Left,
                                VirtualKeyCode::Right if game.snake.direction != Direction::Left => Direction::Right,
                                _ => game.snake.direction,
                            };
                        }
                    }
                }
            }

            Event::RedrawRequested(_) => {
                if last_update.elapsed() >= update_interval {
                    let (_reward, done) = game.update(game.snake.direction);
                    if done {
                        game = Game::new();
                    }
                    last_update = Instant::now();
                }

                draw_frame(pixels.frame_mut(), &game);

                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                }
            }

            Event::MainEventsCleared => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}

pub fn run_best() -> Result<(), String> {
    let save_data = load("snake_model.json").map_err(|_| "Failed to load model for best agent")?;
    let net = Network::from_serializable(save_data.network); // ✅ здесь

    let mut game = Game::new();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake RL - Best Agent")
        .with_inner_size(winit::dpi::LogicalSize::new(
            (game.width * CELL_SIZE as i32) as f64,
            (game.height * CELL_SIZE as i32) as f64,
        ))
        .build(&event_loop)
        .unwrap();

    let surface_texture = SurfaceTexture::new(window.inner_size().width, window.inner_size().height, &window);
    let mut pixels = Pixels::new(game.width as u32, game.height as u32, surface_texture).unwrap();

    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(50);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => {
                if let WindowEvent::CloseRequested = event {
                    *control_flow = ControlFlow::Exit;
                }
            }

            Event::RedrawRequested(_) => {
                if last_update.elapsed() >= update_interval {
                    let state = game.get_state();
                    let q_values = net.predict(&state); // ✅ теперь работает

                    let mut max_idx = 0;
                    let mut max_val = q_values[0];
                    for i in 1..OUTPUT_SIZE {
                        if q_values[i] > max_val {
                            max_val = q_values[i];
                            max_idx = i;
                        }
                    }

                    let best_dir = match max_idx {
                        0 => game.snake.direction.left(),
                        1 => game.snake.direction,
                        2 => game.snake.direction.right(),
                        _ => game.snake.direction,
                    };

                    let (_reward, done) = game.update(best_dir);
                    if done {
                        game = Game::new();
                    }

                    last_update = Instant::now();
                }

                draw_frame(pixels.frame_mut(), &game);

                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                }
            }

            Event::MainEventsCleared => {
                window.request_redraw();
            }

            _ => {}
        }
    });
}
