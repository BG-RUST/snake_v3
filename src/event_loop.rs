use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use pixels::{Pixels, SurfaceTexture};
use std::time::{Duration, Instant};
use crate::game::*;
use crate::snake::*;

// Cell size in pixels for drawing.
const CELL_PX: u32 = 24;

// Default snake step period (ms) for both manual and AI preview.
const STEP_MS_DEFAULT: u64 = 100;

// Manual play in a separate window (arrow keys).
pub fn run_manual(mut game: Game) -> Result<(), String> {
    run_window_loop(game, None)
}

// AI preview in a window (no learning).
// NOTE: agent is passed BY VALUE to satisfy 'static closure requirement of winit.
pub fn run_ai_preview(game: Game, agent: crate::dqn::DQNAgent) -> Result<(), String> {
    run_window_loop(game, Some(agent))
}

// Unified window loop for manual and AI modes.
// If `agent_opt` is Some(agent), we drive the game with the agent; otherwise with arrow keys.
// We OWN agent here, so the 'static closure can freely move it.
fn run_window_loop(mut game: Game, agent_opt: Option<crate::dqn::DQNAgent>) -> Result<(), String> {
    let win_w = (game.width() as u32) * CELL_PX;
    let win_h = (game.height() as u32) * CELL_PX;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(if agent_opt.is_some() { "Snake — AI preview" } else { "Snake — manual" })
        .with_inner_size(LogicalSize::new(win_w as f64, win_h as f64))
        .with_min_inner_size(LogicalSize::new(win_w as f64, win_h as f64))
        .build(&event_loop)
        .map_err(|e| format!("Error creating window: {e}"))?;

    // Fixed-size pixel buffer matches board resolution scaled by CELL_PX.
    let surface = SurfaceTexture::new(win_w, win_h, &window);
    let mut pixels = Pixels::new(win_w, win_h, surface)
        .map_err(|e| format!("Error creating pixels: {e}"))?;

    // Manual control: requested direction from keyboard.
    let mut pending_dir = Dir::Right;

    // Step timing.
    let mut step_ms: u64 = STEP_MS_DEFAULT;
    let mut acc = Duration::ZERO;
    let mut prev = Instant::now();
    let mut paused = false;

    // Simple counters for AI preview.
    let mut ai_return: f32 = 0.0;

    // Make agent_opt mutable and move it into the closure (owned, 'static).
    let mut agent_opt = agent_opt;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                // Draw current game state into the RGBA buffer.
                let frame = pixels.frame_mut();
                render_frame(frame, &game);

                if let Err(e) = pixels.render() {
                    eprintln!("Error rendering frame: {e}");
                    *control_flow = ControlFlow::Exit;
                }
            }

            Event::MainEventsCleared => {
                // Accumulate time elapsed.
                let now = Instant::now();
                acc += now - prev;
                prev = now;

                let step_dt = Duration::from_millis(step_ms);

                // Advance the game logic with a fixed tick.
                while acc >= step_dt {
                    acc -= step_dt;

                    if !paused {
                        if let Some(agent) = agent_opt.as_mut() {
                            // AI-driven step.
                            let obs = game.observe();
                            let a = agent.select_action(&obs);
                            let StepOutcome { reward, done } = game.step_ai(a);
                            ai_return += reward;
                            if done {
                                // Print a short episode line to the console (overlay text is not drawn).
                                println!("AI episode finished | return = {:.3}", ai_return);
                                ai_return = 0.0;
                                game.reset();
                            }
                        } else {
                            // Manual step.
                            game.set_pending_dir(pending_dir);
                            game.step();
                            if game.is_done() {
                                game.reset();
                                pending_dir = Dir::Right;
                            }
                        }
                    }
                }

                // One redraw per frame.
                window.request_redraw();
            }

            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(size) => {
                    if let Err(e) = pixels.resize_surface(size.width, size.height) {
                        eprintln!("Error resizing window: {e}");
                        *control_flow = ControlFlow::Exit;
                    }
                }
                WindowEvent::KeyboardInput {
                    input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(key),
                        ..
                    },
                    ..
                } => {
                    if state == ElementState::Pressed {
                        match key {
                            // Manual arrows (ignored in AI mode, but harmless to set).
                            VirtualKeyCode::Up    => { pending_dir = Dir::Up; }
                            VirtualKeyCode::Down  => { pending_dir = Dir::Down; }
                            VirtualKeyCode::Left  => { pending_dir = Dir::Left; }
                            VirtualKeyCode::Right => { pending_dir = Dir::Right; }

                            // Reset episode.
                            VirtualKeyCode::R => {
                                ai_return = 0.0;
                                game.reset();
                            }

                            // Pause/resume.
                            VirtualKeyCode::Space => {
                                paused = !paused;
                            }

                            // Speed up: '=' or Numpad '+'
                            VirtualKeyCode::Equals | VirtualKeyCode::NumpadAdd => {
                                step_ms = (step_ms.saturating_sub(10)).max(20);
                                println!("speed: {} ms/step", step_ms);
                            }

                            // Slow down: '-' or Numpad '-'
                            VirtualKeyCode::Minus | VirtualKeyCode::NumpadSubtract => {
                                step_ms = (step_ms + 10).min(500);
                                println!("speed: {} ms/step", step_ms);
                            }

                            // Exit.
                            VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            _ => {}
        }
    });
}

// ------- Rendering helpers -------

// Draw one full frame into an RGBA buffer.
fn render_frame(frame: &mut [u8], game: &Game) {
    let win_w = (game.width() as u32) * CELL_PX;
    let win_h = (game.height() as u32) * CELL_PX;

    // Background (dark).
    fill_rect(frame, win_w, win_h, 0, 0, win_w, win_h, [18, 18, 22, 255]);

    // Grid lines.
    for gx in 0..=game.width() as u32 {
        let x = gx * CELL_PX;
        fill_rect(frame, win_w, win_h, x, 0, 1, win_h, [30, 30, 36, 255]);
    }
    for gy in 0..=game.height() as u32 {
        let y = gy * CELL_PX;
        fill_rect(frame, win_w, win_h, 0, y, win_w, 1, [30, 30, 36, 255]);
    }

    // Food (red).
    let (fx, fy) = game.food_pos();
    draw_cell(frame, win_w, win_h, fx as u32, fy as u32, [200, 60, 36, 255]);

    // Snake (green). Head a bit brighter.
    let segs = game.snake_segments();
    for (i, (x, y)) in segs.iter().enumerate() {
        let is_head = i + 1 == segs.len();
        let col = if is_head { [80, 220, 120, 255] } else { [60, 180, 90, 255] };
        draw_cell(frame, win_w, win_h, *x as u32, *y as u32, col);
    }
}

// Draw one cell in grid coordinates.
fn draw_cell(frame: &mut [u8], win_w: u32, win_h: u32, cx: u32, cy: u32, rgba: [u8; 4]) {
    let x0 = cx * CELL_PX;
    let y0 = cy * CELL_PX;
    fill_rect(frame, win_w, win_h, x0, y0, CELL_PX, CELL_PX, rgba);
}

// Fill an axis-aligned rectangle in the pixel buffer (coords in pixels).
fn fill_rect(frame: &mut [u8], win_w: u32, win_h: u32, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    let x1 = (x + w).min(win_w);
    let y1 = (y + h).min(win_h);

    for py in y..y1 {
        let row_off = (py as usize) * (win_w as usize) * 4;
        for px in x..x1 {
            let off = row_off + (px as usize) * 4;
            frame[off + 0] = rgba[0]; // R
            frame[off + 1] = rgba[1]; // G
            frame[off + 2] = rgba[2]; // B
            frame[off + 3] = rgba[3]; // A
        }
    }
}
