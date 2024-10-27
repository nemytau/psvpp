// distance_manager.rs

use std::f64::consts::PI;
use crate::structs::node::Node;

// Constant for Earth’s radius in nautical miles
const EARTH_RADIUS_NM: f64 = 3440.0; // nautical miles

// Helper function using the Haversine formula for distance calculation in nautical miles
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1_rad = lat1 * PI / 180.0;
    let lat2_rad = lat2 * PI / 180.0;
    let delta_lat = (lat2 - lat1) * PI / 180.0;
    let delta_lon = (lon2 - lon1) * PI / 180.0;

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_NM * c
}

#[derive(Debug)]
pub struct DistanceManager {
    distances: Vec<Vec<f64>>, // 2D matrix to store distances in nautical miles
}

impl DistanceManager {
    // Constructor: initializes an empty matrix for distances of size n x n
    pub fn new(num_nodes: usize) -> Self {
        let distances = vec![vec![0.0; num_nodes]; num_nodes];
        DistanceManager { distances }
    }

    // Populates the distance matrix using the coordinates of nodes
    pub fn calculate_distances(&mut self, nodes: &[Node]) {
        let num_nodes = nodes.len();
        for i in 0..num_nodes {
            for j in i + 1..num_nodes {
                // Calculate the distance in nautical miles using Haversine formula
                let dist = haversine_distance(
                    nodes[i].location.latitude,
                    nodes[i].location.longitude,
                    nodes[j].location.latitude,
                    nodes[j].location.longitude,
                );
                // Set symmetric values in the distance matrix
                self.distances[i][j] = dist;
                self.distances[j][i] = dist;
            }
        }
    }

    // Retrieves the distance between two nodes by their IDs
    pub fn distance(&self, from: usize, to: usize) -> f64 {
        self.distances[from][to]
    }
}