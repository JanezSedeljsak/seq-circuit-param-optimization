extern crate nannou;
mod util;

use std::path::Path;
use nannou::prelude::*;

const NUM_ROWS: usize = 2;
const NUM_COLS: usize = 5;

const POPULATION: usize = 10;
const GENERATIONS: usize = 10;

const FILE: &str = "data/export_GreyWolfOptimizer.csv";

struct State {
    graphs: Box<[[[[f32; 100]; 3]; POPULATION]; GENERATIONS]>,
    transition: Box<[[[f32; 100]; 3]; POPULATION]>,
    transition_index: usize,
    generation_index: usize,
    best_index: i32,
    max_value: f32,
}

fn main() {
    nannou::app(state)
        .update(update)
        .simple_window(view)
        .size(1280, 480)
        .run();
}

fn state(_app: &App) -> State {
    let file_path = Path::new(FILE);
    let transition = Box::new([[[0.0; 100]; 3]; POPULATION]);
    let transition_index = 0;
    let generation_index = 0;
    let best_index = -1;
    let mut graphs = Box::new([[[[0.0; 100]; 3]; POPULATION]; GENERATIONS]);

    if let Ok(lines) = util::read_lines(file_path) {
        for (idx, line) in lines.enumerate() {
            if let Ok(row) = line {
                if let Err(err) = util::parse_generation(&row, &mut graphs, idx) {
                    println!("Error: {}", err);
                }
            }
        }
    }

    let max_value = 150.0;
    State { graphs, transition, transition_index, generation_index, best_index, max_value }
}

fn update(_app: &App, state: &mut State, _update: Update) {
    if state.generation_index < GENERATIONS - 1 {
        state.transition_index += 1;

        // generate new transitions
        if state.transition_index >= 100 {
            state.transition_index = 0;
            state.generation_index += 1;
            for row in 0..NUM_ROWS {
                for col in 0..NUM_COLS {
                    let graph_index = row * NUM_COLS + col;
                    util::fill_transition(
                        &mut state.transition[graph_index],
                        &state.graphs[state.generation_index - 1][graph_index],
                        &state.graphs[state.generation_index][graph_index]
                    );
                }
            }
        }
    }
}

fn view(_app: &App, state: &State, frame: Frame) {
    let draw = _app.draw();
    draw.background().color(BLACK);
  
    let cell_width = _app.window_rect().w() / NUM_COLS as f32;
    let cell_height = _app.window_rect().h() / NUM_ROWS as f32;

    let x_scale = (cell_width - 10.0) / 100.0;
    let y_scale = (cell_height - 10.0) / state.max_value;

    for row in 0..NUM_ROWS {
        for col in 0..NUM_COLS {
            let graph_index = row * NUM_COLS + col;
       
            let x = col as f32 * cell_width - _app.window_rect().w() / 2.0 + cell_width / 2.0;
            let y = row as f32 * cell_height - _app.window_rect().h() / 2.0 + cell_height / 2.0;
            let rect = Rect::from_x_y_w_h(x, y, cell_width, cell_height);
            draw.rect().xy(rect.xy()).wh(rect.wh()).color(BLACK).stroke(DARKGRAY).stroke_weight(2.0);
            
            let horizontal_offset = x - cell_width / 2.0 + 5.0;
            let vertical_offset = y - cell_height / 2.0 + 5.0;
            let colors = [RED, BLUE, GREEN];

            let text = format!("Graph {}", graph_index);
            let text_position = pt2(x, y + cell_width / 3.0);
            let text_color = if state.best_index == graph_index as i32 { GREEN } else { DARKGRAY };

            draw.text(&text)
                .xy(text_position)
                .color(text_color)
                .font_size(20);
            
            for j in 0..3 {
                let points = (0..100).map(|i| {
                    let x_itt = i as f32 * x_scale + horizontal_offset;
                    let mut y_itt = state.graphs[state.generation_index][graph_index][j][i];
                    if !y_itt.is_finite() {
                        y_itt = 0.0;
                    }
                    
                    let delta = state.transition[graph_index][j][i];
                    y_itt += (if delta.is_finite() { delta } else { 0.0 }) * state.transition_index as f32;
                    if y_itt < 0.0 {
                        y_itt = 0.0;
                    }

                    y_itt = y_itt * y_scale + vertical_offset;
                    (pt2(x_itt, y_itt), colors[j])
                });
                draw.polyline().points_colored(points);
            }
        }
    }

    draw.to_frame(_app, &frame).unwrap();
}

