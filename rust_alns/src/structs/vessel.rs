use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
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

impl Vessel {
    pub fn new(
        idx: u32,
        name: String,
        deck_capacity: u32,
        bulk_capacity: u32,
        speed: f64,
        vessel_type: String,
        fcs: f64,
        fcw: f64,
        cost: u32,
    ) -> Self {
        Self {
            idx,
            name,
            deck_capacity,
            bulk_capacity,
            speed,
            vessel_type,
            fcs,
            fcw,
            cost,
        }
    }

    pub fn get_idx(&self) -> u32 {
        self.idx
    }

}