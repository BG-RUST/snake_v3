use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};
use pixels::{Pixels, SurfaceTexture};
use std::time::{Instant, Duration};

use crate::game::*;
use crate::game_input::GameInput;
use crate::brain::Brain;
use crate::db::*;

pub const GRID_WIDTH: usize = 10;
pub const GRID_HEIGHT: usize = 10;
const SCALE: u32 = 32;
const MAX_STEPS: usize = 1000;


pub fn run_brain_play() {
    //let model = load_brain().expect("Failed to load brain from file");
    let checkpoint = load_checkpoint("best_model.json").unwrap_or_else(|| {
        eprintln!("❌ Не удалось загрузить best_model.json. Запустите обучение с --train.");
        std::process::exit(1);
    });

    println!("✅ Загружена модель с reward = {:.8}", checkpoint.meta.average_reward);

    let brain = Brain::from_model(&checkpoint.model);


    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake AI - Brain Playback")
        .with_inner_size(LogicalSize::new(
            GRID_WIDTH as u32 * SCALE,
            GRID_HEIGHT as u32 * SCALE,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let surface = SurfaceTexture::new(GRID_WIDTH as u32 * SCALE, GRID_HEIGHT as u32 * SCALE, &window);
    let mut pixels = Pixels::new(GRID_WIDTH as u32, GRID_HEIGHT as u32, surface).unwrap();

    let mut game = Game::new(GRID_WIDTH, GRID_HEIGHT);
    let mut steps = 0;
    let start_pos = game.snake().head();
    let mut steps_since_eat = 0;
    let update_interval = Duration::from_millis(100);
    let mut last_update = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                if last_update.elapsed() >= update_interval {
                    let input = GameInput::from_game(&game, 0, start_pos, steps_since_eat);
                    let action = brain.predict(&input);
                    game.set_action_index(action);

                    game.update();
                    steps += 1;
                    steps_since_eat += 1;
                    last_update = Instant::now();

                    if game.snake().head() == game.food() {
                        steps_since_eat = 0;
                    }

                    if game.snake().is_dead(GRID_WIDTH, GRID_HEIGHT) || steps > MAX_STEPS {
                        println!("🛑 DEAD or MAX STEPS reached");
                        *control_flow = ControlFlow::Exit;
                    }
                }

                draw(pixels.frame_mut(), &game);
                pixels.render().unwrap();
            }

            Event::WindowEvent { event, .. } => {
                if let WindowEvent::CloseRequested = event {
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

fn draw(frame: &mut [u8], game: &Game) {
    let width = game.width() as i32;

    for pixel in frame.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0x10, 0x10, 0x10, 0xFF]);
    }

    let food = game.food();
    let i = (food.y * width + food.x) * 4;
    if i >= 0 && (i as usize) + 3 < frame.len() {
        let i = i as usize;
        frame[i + 0] = 0xFF;
        frame[i + 1] = 0x00;
        frame[i + 2] = 0x00;
        frame[i + 3] = 0xFF;
    }

    for seg in game.snake().body() {
        let i = (seg.y * width + seg.x) * 4;
        if i >= 0 && (i as usize) + 3 < frame.len() {
            let i = i as usize;
            frame[i + 0] = 0x00;
            frame[i + 1] = 0xFF;
            frame[i + 2] = 0x00;
            frame[i + 3] = 0xFF;
        }
    }
}


pub fn run_human_play() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake - Игрок")
        .with_inner_size(LogicalSize::new(
            GRID_WIDTH as u32 * SCALE,
            GRID_HEIGHT as u32 * SCALE,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let surface = SurfaceTexture::new(GRID_WIDTH as u32 * SCALE, GRID_HEIGHT as u32 * SCALE, &window);
    let mut pixels = Pixels::new(GRID_WIDTH as u32, GRID_HEIGHT as u32, surface).unwrap();

    let mut game = Game::new(GRID_WIDTH, GRID_HEIGHT);
    let update_interval = Duration::from_millis(100);
    let mut last_update = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                if last_update.elapsed() >= update_interval {
                    game.update();
                    last_update = Instant::now();
                }

                draw(pixels.frame_mut(), &game);
                pixels.render().unwrap();
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(key) = input.virtual_keycode {
                        if let Some(dir) = game.key_to_direction(key) {
                            game.set_direction(dir);
                        }
                    }
                }
                _ => {}
            },

            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}