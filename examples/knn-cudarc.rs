// Author: Ratnodeep Bandyopadhyay
// All rights reserved 2023.

use std::fs::File;
use std::io::{BufRead, BufReader};

// HERE
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
// HERE

// Define the Feature Size
// SMALL_TEST
// const FEATURE_SIZE: usize = 2;
// const DATASET_SIZE: usize = 7;
// const TESTSET_SIZE: usize = 4;
// BIG TEST
const FEATURE_SIZE: usize = 13;
const DATASET_SIZE: usize = 255;
const TESTSET_SIZE: usize = 48;

const K:            usize = 50;
const TOTAL_CLASS:  usize = 2;

// HERE
const PTX_SRC: &str = "
extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
    int i = threadIdx.x;
    float temp = A[i] - B[i];
    C[i] = temp * temp;
}
";
// HERE


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
    let mut class_counter: [u32; TOTAL_CLASS] = [0; TOTAL_CLASS];

    for i in 0..DATASET_SIZE {
        let mut replacement_idx: usize = 0;

        let test_distance_point = PointDistance {
            point: Some(&dataset[i]),
            distance: Some(
                euclidian(target, &dataset[i])
            ),
        };


        'inner: for i in 0..K {
            if selected_points[i].distance.is_none() {
                replacement_idx = i;
                break 'inner;
            } else if selected_points[i].distance.is_some() {
                if selected_points[i].distance.unwrap() >= 
                selected_points[replacement_idx].distance.unwrap() {
                    replacement_idx = i;
                }
            }
        }

        if selected_points[replacement_idx].distance.is_none() {
            selected_points[replacement_idx] = test_distance_point;
        }
        // replace old min with new min
        else if test_distance_point.distance.unwrap() <= 
        selected_points[replacement_idx].distance.unwrap() {
            selected_points[replacement_idx] = test_distance_point;
        }
    }

    for point in selected_points {
        class_counter[point.point.unwrap().class.unwrap() as usize] += 1;
    }

    // print!("selected_points: [");
    // for point in selected_points {
    //     print!("{:?}, ", point.point.unwrap().point);
    // }
    // println!("]");
    // println!("{:?}", target.point);
    // println!("class_counter: {:?}\n", class_counter);

    let mut closest_class: usize = 0;
    for (class_idx, class_count) in class_counter.iter().enumerate() {
        if class_counter[closest_class as usize] <= *class_count {
            closest_class = class_idx;
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


    // let a_host = [1.0f32, 2.0, 3.0, 4.0];
    // let b_host = [1.0f32, 2.0, 3.0, 4.0];
    let a_host = a.point
        .iter()
        .filter_map(|point_option| point_option.clone())
        .collect();
    let b_host = b.point
        .iter()
        .filter_map(|point_option| point_option.clone())
        .collect();
    let mut c_host = [0.0f32; K];

    let a_dev = dev.htod_sync_copy(&a_host).unwrap();
    let b_dev = dev.htod_sync_copy(&b_host).unwrap();
    let mut c_dev = dev.htod_sync_copy(&c_host).unwrap();

    // println!("Copied in {:?}", start.elapsed());

    let cfg = LaunchConfig {
        block_dim: (1, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { f.launch(cfg, (&a_dev, &b_dev, &mut c_dev, K)) }.unwrap();

    dev.dtoh_sync_copy_into(&c_dev, &mut c_host).unwrap();

    c_host.iter().sum()

    // let mut sum: f32 = 0.0;
    // let mut temp: f32;
    // for i in 0..FEATURE_SIZE {
    //     temp = a.point[i].unwrap() - b.point[i].unwrap();
    //     sum += temp * temp;
    // }
    // sum
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
            if val_idx == 0 {
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
            if val_idx == 0 {
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

fn main() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx(PTX_SRC).unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let dev = CudaDevice::new(0)?;
    println!("Built in {:?}", start.elapsed());

    dev.load_ptx(ptx, "matmul", &["matmul"])?;
    let f = dev.get_func("matmul", "matmul").unwrap();
    println!("Loaded in {:?}", start.elapsed());
    // let path_to_dataset = "/Users/rb/School/CS217/knn_cuda/assets/sample_train.csv";
    // let path_to_testset = "/Users/rb/School/CS217/knn_cuda/assets/sample_test.csv";
    let path_to_dataset = "/home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/heart_data_norm.csv";
    let path_to_testset = "/home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/heart_data_norm_test.csv";
    ///home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/heart_data_norm.csv
    ///home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/heart_data_norm.csv.back
    ///home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/heart_data_norm_test.csv
    ///home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/sample_test.csv
    ///home/cemaj/rbandyopadhyay/cs217/final-proj/cudarc/assets/sample_train.csv
    // Load dataset
    let dataset = load_dataset(path_to_dataset);
    // Load testvector set
    let testset = load_testset(path_to_testset);
    // perform knn on testset
    let guesses = knn_vec(&testset, &dataset);
    // evaluate the accuracy
    let accuracy = evalulate_accuracy(&testset, &guesses);
    println!("Accuracy: {}", accuracy);
    Ok(())
}
