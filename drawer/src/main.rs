extern crate nannou;
use nannou::prelude::*;

struct Model {
    y_values: [f32; 200],
}

fn main() {
    nannou::app(model)
        .update(update)
        .simple_window(view)
        .size(1280, 480)
        .run();
}

fn model(_app: &App) -> Model {
    let mut y_values = [0.0; 200];
    for i in 0..200 {
        y_values[i] = i as f32;
    }

    Model { y_values }
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    // Update the y values array here if needed
}

fn view(_app: &App, model: &Model, frame: Frame) {
    let draw = _app.draw();
    draw.background().color(DARKGRAY);

    let num_rows = 2;
    let num_cols = 5;
    let cell_width = _app.window_rect().w() / num_cols as f32;
    let cell_height = _app.window_rect().h() / num_rows as f32;

    for row in 0..num_rows {
        for col in 0..num_cols {
            let x = col as f32 * cell_width - _app.window_rect().w() / 2.0 + cell_width / 2.0;
            let y = row as f32 * cell_height - _app.window_rect().h() / 2.0 + cell_height / 2.0;
            let rect = Rect::from_x_y_w_h(x, y, cell_width, cell_height);
            draw.rect().xy(rect.xy()).wh(rect.wh()).color(WHITE).stroke(BLACK).stroke_weight(2.0);

            let points = (0..200).map(|i| {
                let x_itt = i as f32 + x - cell_width / 2.0;
                let y_itt = model.y_values[i] + y - cell_height / 2.0 + 5.0;
                // println!("Point {}: x={}, y={}", i, x, y); // Print the points to the terminal
                (pt2(x_itt, y_itt), BLACK)
            });
            draw.polyline().points_colored(points);
            
        }
    }

    draw.to_frame(_app, &frame).unwrap();
    
}
