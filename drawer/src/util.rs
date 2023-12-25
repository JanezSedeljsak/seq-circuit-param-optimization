use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::error::Error;
use crate::constants::*;

pub fn read_lines<P>(file_path: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(file_path)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn parse_generation(
    input: &str, graphs: &mut Box<[[[[f32; LINE_PERCISION]; 3]; POPULATION]; GENERATIONS]>, 
    index: usize
) -> Result<(), Box<dyn Error>> {
    if index < GENERATIONS {
        let generation: Vec<f32> = input
            .split(',')
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();

        for itt in 0..POPULATION {
            let start = itt * 3000;
            let end = start + 3000;

            let individual = &generation[start..end];
            let minified_individual: Vec<f32> = individual
                .chunks(CHUNKS_SCALE as usize)
                .map(|chunk| {
                    let sum: f32 = chunk.iter().sum();
                    sum / CHUNKS_SCALE
                })
                .collect();
            
            for (i, chunk) in minified_individual.chunks(LINE_PERCISION).enumerate().take(3) {
                graphs[index][itt][i].copy_from_slice(chunk);
            }
        }

        Ok(())
    } else {
        Err("Index out of bounds".into())
    }
}

pub fn fill_transition(
    transition: &mut [[f32; LINE_PERCISION]; 3], start: &[[f32; LINE_PERCISION]; 3], 
    end: &[[f32; LINE_PERCISION]; 3], transitions_count: usize
) {
    for j in 0..3 {
        for i in 0..LINE_PERCISION {
            let diff = end[j][i] - start[j][i];
            transition[j][i] = diff / transitions_count as f32;
        }
    }
}

pub fn reset_transition(transition: &mut [[[f32; LINE_PERCISION]; 3]; POPULATION])
{
    for i in 0..POPULATION {
        for j in 0..3 {
            for k in 0..LINE_PERCISION {
                transition[i][j][k] = 0.0;
            }
        }
    }
}


pub fn load_clk(clk: &mut [f32; LINE_PERCISION]) -> Result<(), Box<dyn Error>> {
    let mut unparsed_clk: [f32; 1000] = [0.0; 1000];
    if let Ok(lines) = read_lines(CLK_FILE) {
        for (idx, line) in lines.enumerate() {
            if let Ok(row) = line {
                unparsed_clk[idx] = row.parse().unwrap_or(0.0);
            } else {
                return Err("Error while parsing line".into())
            }
        }
    } else {
        return Err("Error while reading lines".into())
    }

    let clk_minified: Vec<f32> = unparsed_clk
        .chunks(CHUNKS_SCALE as usize)
        .map(|chunk| {
            let sum: f32 = chunk.iter().sum();
            sum / CHUNKS_SCALE
        })
        .collect();
    
    for i in 0..LINE_PERCISION {
        clk[i] = clk_minified[i];
    }

    Ok(())
}

