use winit::{
    application::ApplicationHandler,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop, ActiveEventLoop},
    window::{Window, WindowId},
    dpi::LogicalSize,
};

#[derive(Default)]
struct App {
    window: Option<Window>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop){
        let window_attributes = Window::default_attributes()
            .with_title("Snake + deep learning on Rust")
            .with_inner_size(LogicalSize::new(640.0, 640.0));

        let window = event_loop.create_window(window_attributes).unwrap();
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: WindowId, event: WindowEvent){
        match event {
            WindowEvent::CloseRequested => {
                println!("Closing");
                event_loop.exit();
            }
            _ => {}
        }
    }
}

pub fn start() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}