use winit::{
    dpi::LogicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent },
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};
use pixels::{Pixels, SurfaceTexture};
use std::time::{Duration, Instant};
use crate::game::*;
use crate::snake::*;

//cell size in pixels
const CELL_PX: u32 = 24;
//speed snake
const STEP_MS:u64 = 100;

//manual game mod in separate window
pub fn run_manual(mut game: Game) -> Result<(), String> {
    let win_w = (game.width() as u32) * CELL_PX;
    let win_h = (game.height() as u32) * CELL_PX;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake - manual")
        .with_inner_size(LogicalSize::new(win_w as f64, win_h as f64))
        .with_min_inner_size(LogicalSize::new(win_w as f64, win_h as f64))
        .build(&event_loop)
        .map_err(|e| format!("Error creating window: {}", e))?;

    //surface texture for pixels
    let surface = SurfaceTexture::new(win_w, win_h, &window);
    //initialization pixels with fix buffer
    let mut pixels = Pixels::new(win_w, win_h, surface)
        .map_err(|e| format!("Error creating pixels: {}", e))?;

    //requested direction from the keyboard, passed to the game on each tick
    let mut pending_dir = Dir::Right;

    //logiacal update timer
    let mut acc = Duration::ZERO;
    let step_dt = Duration::from_millis(STEP_MS);
    let mut prev = Instant::now();

    //run event loop
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::RedrawRequested(_) => {
                //get a mutable slice of the RGBA buffer
                let frame = pixels.frame_mut();
                //draw current game state into the buffer
                render_frame(frame, &game);

                //display the frame on thw screen
                if let Err(e) = pixels.render() {
                    eprintln!("Error rendering frame: {}", e);
                    *control_flow = ControlFlow::Exit;
                }
            }
            //the main point of the loop is to check whether it is time to take a logical step
            Event::MainEventsCleared => {
                let now = Instant::now();
                acc += now - prev;
                prev = now;

                while acc >= step_dt {
                    acc -= step_dt;

                    game.set_pending_dir(pending_dir); // применяем 1 раз на тик
                    game.step();

                    if game.is_done() {
                        game.reset();
                        pending_dir = Dir::Right;
                    }
                }

                // один redraw на кадр
                window.request_redraw();
            }

            //handle window/keyboard events
            Event::WindowEvent {event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(size) => {
                    //adjust the surface to the new window (the buffer remains the same size)
                    if let Err(e) = pixels.resize_surface(size.width, size.height) {
                        eprintln!("Error resizing window: {}", e);
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
                            VirtualKeyCode::Up => pending_dir = Dir::Up,
                            VirtualKeyCode::Down => pending_dir = Dir::Down,
                            VirtualKeyCode::Left => pending_dir = Dir::Left,
                            VirtualKeyCode::Right => pending_dir = Dir::Right,
                            VirtualKeyCode::R => game.reset(),
                            VirtualKeyCode::Escape => *control_flow = ControlFlow::Exit,
                            _ => {}
                        }
                    }
                }
                _=> {}
            },
            _ => {}
        }

    });
}
//helper function: draw one frame to an RGBA buffer.
fn render_frame(frame: &mut [u8], game: &Game) {
    let win_w = (game.width() as u32) * CELL_PX;
    let win_h = (game.height() as u32) * CELL_PX;

    //Background fill (dark)
    fill_rect(frame, win_w, win_h, 0, 0, win_w, win_h, [18, 18, 22, 255]);

    //draw the grid with thin lines
    for gx in 0..=game.width() as u32 {
        let x = gx * CELL_PX;
        fill_rect(frame, win_w, win_h, x, 0, 1, win_h, [30, 30, 36, 255]);
    }
    for gy in 0..=game.height() as u32 {
        let y = gy * CELL_PX;
        fill_rect(frame, win_w, win_h, 0,y, win_w, 1, [30, 30, 36, 255]);
    }
    //food - red square
    let (fx, fy) = game.food_pos();
    draw_cell(frame, win_w, win_h, fx as u32, fy as u32, [200, 60, 36, 255]);

    //snake body is green squares
    //head is lighter for distinction

    let segs = game.snake_segments();
    for (i, (x, y)) in segs.iter().enumerate() {
        let is_head = i + 1 == segs.len();
        let col = if is_head { [80, 220, 120, 255] } else { [60, 180, 90, 255] };
        draw_cell(frame, win_w, win_h, *x as u32, *y as u32, col);
    }
}

//draw one cell
fn draw_cell(frame: &mut [u8], win_w: u32, win_h: u32, cx: u32, cy: u32, rgba: [u8; 4]) {
    let x0 = cx * CELL_PX;
    let y0 = cy * CELL_PX;
    fill_rect(frame, win_w, win_h, x0, y0, CELL_PX, CELL_PX, rgba);
}

//fill the rectangle in the pixel buffer, coordinates in pixels
fn fill_rect(frame: &mut [u8], win_w: u32, win_h: u32, x: u32, y: u32, w: u32, h: u32, rgba: [u8; 4]) {
    // Ограничиваем прямоугольник границами буфера.
    //limit rectangle to the buffer boundaries
    let x1 = (x + w).min(win_w);
    let y1 = (y + h).min(win_h);

    //we go through the rows and columns of the ractangle
    for py in y..y1 {
        //offset the start of the line in bytes (4 bytes on pixel)
        let row_off = (py as usize) * (win_w as usize) * 4;
        for px in x..x1 {
            //offset of a specific pixel (offset - смещение)
            let off = row_off + (px as usize) * 4;
            frame[off + 0] = rgba[0]; // R
            frame[off + 1] = rgba[1]; // G
            frame[off + 2] = rgba[2]; // B
            frame[off + 3] = rgba[3]; // A

        }
    }
}
