// Author: Ratnodeep Bandyopadhyay
// All rights reserved 2023.
//
use std::fs::File;
use std::io::{BufRead, BufReader};

// Define the Feature Size
const FEATURE_SIZE: usize = 5;
const DATASET_SIZE: usize = 5;
const TESTSET_SIZE: usize = 1;
const K:            usize = 4;

// Point structure
#[derive(Clone, Copy)]
struct Point32 {
    point: [Option<f32>; FEATURE_SIZE],
    class: Option<u32>,
}
// renaming struct to something simpler
type p32 = Point32;
// implementations of Point32
impl Point32 {
    fn new() -> Point32 {
        Point32 {
            point: [None; FEATURE_SIZE],
            class: None,
        }
    }
}

// Point structure
// #[derive(Clone, Copy)]
// struct Point64 {
//     point: [Option<f64>; FEATURE_SIZE],
//     class: Option<u32>,
// }
// // renaming struct to something simpler
// type p64 = Point64;

#[derive(Clone, Copy)]
struct PointDistance<'a> {
    point: Option<&'a Point32>,
    distance: Option<f32>,
}

// implementations
impl PointDistance<'_> {
    fn new() -> PointDistance<'static> {
        PointDistance {
            point: None,
            distance: None,
        }
    }
}


fn knn(target: &p32, dataset: &[p32; DATASET_SIZE]) -> p32 {

    let mut selected_points: [PointDistance; K] = [PointDistance::new(); K];
    let mut class_counter: [u32; K] = [0; K];

    for i in 0..DATASET_SIZE {
        let mut replacement_idx: Option<usize> = None;

        let test_distance_point = PointDistance {
            point: Some(&dataset[i]),
            distance: Some(
                euclidian(target, &dataset[i])
            ),
        };


        // `get_idx: for point in selected_points {
        'inner: for i in 0..K {
            match selected_points[i].distance {
                None => {
                    replacement_idx = Some(i);
                    break 'inner;
                },
                Some(distance) => {
                    if distance >= test_distance_point.distance.unwrap() {
                        replacement_idx = Some(i);
                    } else {
                        // change nothing.
                    }
                }
            }
        }

        match replacement_idx {
            None => {/* no min was found */},
            Some(idx) => {
                // replace old min with new min
                selected_points[idx] = test_distance_point;
            },
        }
    }

    for point in selected_points {
        class_counter[point.point.unwrap().class.unwrap() as usize] += 1;
    }

    let mut closest_class: u32 = 0;
    for class in class_counter {
        if class_counter[closest_class as usize] <= 
        class_counter[class as usize] {
            closest_class = class;
        }
    }

    Point32 {
        point: target.point.clone(),
        class: Some(closest_class as u32),
    }
}


// Iterate through each point
fn knn_vec(target: &[p32; TESTSET_SIZE],
    dataset: &[p32; DATASET_SIZE]) -> [p32; TESTSET_SIZE] {

    let mut guess: [p32; TESTSET_SIZE] = [p32::new(); TESTSET_SIZE];

    for i in 0..TESTSET_SIZE {
        guess[i] = knn(&target[i], dataset);
    }

    return guess;

}

// Returns the euclidian distance between two points a and b.
fn euclidian(a: &p32, b: &p32) -> f32 {
    let mut sum: f32 = 0.0;
    let mut temp: f32;
    for i in 0..FEATURE_SIZE {
        temp = a.point[i].unwrap() - b.point[i].unwrap();
        sum += temp * temp;
    }
    sum
}

fn evalulate_accuracy(target: &[p32; TESTSET_SIZE],
    guess: &[p32; TESTSET_SIZE]) -> f32 {

    let mut match_sum: f32 = 0.0;

    for i in 0..TESTSET_SIZE {
        if target[i].class == guess[i].class {
            match_sum += 1.0;
        }
    }

    match_sum / (TESTSET_SIZE as f32)
}


fn load_testset(path: &str) -> [p32; TESTSET_SIZE] {
    // Open the file
    // File handler
    let file = File::open(path).unwrap();
    // Create a buffered reader for efficiency
    let reader = BufReader::new(file);

    
    let mut testset: [p32; TESTSET_SIZE] = [p32::new(); TESTSET_SIZE];

    // Read and parse each line of the CSV file
    for (pnt_idx, point) in reader.lines().enumerate() {
        for (val_idx, value) in point.unwrap().split(',').enumerate() {
            if val_idx == 1 {
                match value.trim().parse::<u32>() {
                    Ok(value) => {
                        testset[pnt_idx].class = Some(value);
                    }
                    Err(_) => println!("Value could not be parsed!"),
                }
            } else {
                match value.trim().parse::<f32>() {
                    Ok(value) => testset[pnt_idx].point[val_idx-1] = Some(value),
                    Err(_) => println!("Value could not be parsed!"),
                }
            }
        }
    }

    testset
}

fn load_dataset(path: &str) -> [p32; DATASET_SIZE] {
    // Open the file
    // File handler
    let file = File::open(path).unwrap();
    // Create a buffered reader for efficiency
    let reader = BufReader::new(file);

    let mut dataset: [p32; DATASET_SIZE] = [p32::new(); DATASET_SIZE];

    // Read and parse each line of the CSV file
    for (pnt_idx, point) in reader.lines().enumerate() {
        for (val_idx, value) in point.unwrap().split(',').enumerate() {
            if val_idx == 1 {
                match value.trim().parse::<u32>() {
                    Ok(value) => {
                        dataset[pnt_idx].class = Some(value);
                    }
                    Err(_) => println!("Value could not be parsed!"),
                }
            } else {
                match value.trim().parse::<f32>() {
                    Ok(value) => dataset[pnt_idx].point[val_idx-1] = Some(value),
                    Err(_) => println!("Value could not be parsed!"),
                }
            }
        }
    }

    dataset 
}

fn main() {
    let path_to_dataset = "";
    let path_to_testset = "";
    // Load dataset
    let dataset = load_dataset(path_to_dataset);
    // Load testvector set
    let testset = load_testset(path_to_testset);
    // perform knn on testset
    let guesses = knn_vec(&testset, &dataset);
    // evaluate the accuracy
    let accuracy = evalulate_accuracy(&testset, &guesses);
    println!("Accuracy: {}", accuracy);
}
