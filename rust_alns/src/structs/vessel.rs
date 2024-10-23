use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Vessel {
    pub idx: u32,                 
    pub name: String,
    pub deck_capacity: u32,
    pub bulk_capacity: u32,
    pub speed: f64,
    pub vessel_type: String,
    pub fcs: f64,                 // Fuel consumption sailing
    pub fcw: f64,                 // Fuel consumption waiting
    pub cost: u32,
}