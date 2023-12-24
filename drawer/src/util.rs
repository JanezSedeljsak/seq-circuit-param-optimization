use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::error::Error;

pub fn read_lines<P>(file_path: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(file_path)?;
    Ok(io::BufReader::new(file).lines())
}

pub fn parse_generation(input: &str, graphs: &mut Box<[[[[f32; 100]; 3]; 10]; 10]>, index: usize) -> Result<(), Box<dyn Error>> {
    if index < 10 {
        let generation: Vec<f32> = input
            .split(',')
            .map(|s| s.parse().unwrap_or(0.0))
            .collect();

        for itt in 0..10 {
            let start = itt * 3000;
            let end = start + 3000;

            let individual = &generation[start..end];
            let minified_individual: Vec<f32> = individual
                .chunks(10)
                .map(|chunk| {
                    let sum: f32 = chunk.iter().sum();
                    sum / 10.0
                })
                .collect();
            
            for (i, chunk) in minified_individual.chunks(100).enumerate().take(3) {
                graphs[index][itt][i].copy_from_slice(chunk);
            }
        }

        Ok(())
    } else {
        Err("Index out of bounds".into())
    }
}

pub fn fill_transition(transition: &mut [[f32; 100]; 3], start: &[[f32; 100]; 3], end: &[[f32; 100]; 3]) {
    for j in 0..3 {
        for i in 0..100 {
            let diff = end[j][i] - start[j][i];
            transition[j][i] = diff / 99.0;
        }
    }
}