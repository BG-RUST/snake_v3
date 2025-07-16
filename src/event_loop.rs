use winit::{
    event::{Event, WindowEvent, KeyboardInput, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
    dpi::LogicalSize,
};
use pixels::{Pixels, SurfaceTexture};
use std::time::{ Instant, Duration};

use crate::game::Game;

/// game size in "cells" no pixels
/// real size depends on scale
const GRID_WIDTH: usize = 20;
const GRID_HEIGHT: usize = 20;
const scale: u32 = 32; // every "tail" will be 32x32 pixels

pub fn run() {
    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(150);
    //create window and event loop
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Snake with ai alfa version 4.0")
        .with_inner_size(LogicalSize::new(
            (GRID_WIDTH as u32) * scale,
            (GRID_HEIGHT as u32) * scale,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    //rendering buffer through Pixels
    let surface = SurfaceTexture::new(
        (GRID_WIDTH as u32) * scale,
        (GRID_HEIGHT as u32) * scale,
        &window,
    );
    let mut pixels = Pixels::new(GRID_WIDTH as u32, GRID_HEIGHT as u32, surface).unwrap();

    //game state initialization
    let mut game = Game::new(GRID_WIDTH, GRID_HEIGHT);

    //main event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            //redraw
            Event::RedrawRequested(_) => {
                if last_update.elapsed() >= update_interval {
                    game.update();
                    last_update = Instant::now();
                }
                draw(pixels.frame_mut(), &game);
                if pixels.render().is_err() {
                    *control_flow = ControlFlow::Exit;
                }
            }

            //window event handling
            Event::WindowEvent { event, ..} => match event {
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

            //system update (redraw every tick)
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}
//simple drawing of cells (background snake food)
fn draw(frame: &mut [u8], game: &Game) {
    let width = game.width();
    let height = game.height();

    //clear background
    for pixel in frame.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[0x10, 0x10, 0x10, 0xFF]); //dark-grey
    }

    //food draw
    let food = game.food();
    let i = (food.y * width + food.x) * 4;
    if i < frame.len() {
        frame[i+0] = 0xFF; //red
        frame[i+1] = 0x00;
        frame[i+2] = 0x00;
        frame[i+3] = 0xFF;
    }

    //snake draw
    for seg in game.snake().body() {
        let i = (seg.y * width + seg.x) * 4;
        if i < frame.len() {
            frame[i+0] = 0x00;//gren
            frame[i+1] = 0xFF;
            frame[i+2] = 0x00;
            frame[i+3] = 0xFF;
        }
    }
}