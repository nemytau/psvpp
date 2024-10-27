use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use serde::Deserialize;
use crate::structs::node::{Installation, Location, TimeWindow, Base};
use std::str::FromStr;
use serde::de::DeserializeOwned;

#[derive(Debug, Deserialize)]
pub struct InstallationCSV {
    idx: u32,
    name: String,
    inst_type: String,
    deck_demand: u32,
    visit_frequency: u32,
    #[serde(deserialize_with = "parse_location")]
    location: (f64, f64),  // Parsed as tuple from "[latitude, longitude]"
    departure_spread: u32,
    #[serde(skip_deserializing)] 
    _deck_service_speed: u32,
    #[serde(deserialize_with = "parse_time_window")]
    time_window: (u32, u32),  // Parsed as tuple from "(start, end)"
    service_time: f64,  // Changed to f64 to handle the decimal values
}

impl InstallationCSV {
    // Convert InstallationCSV to Installation object
    pub fn to_installation(self) -> Installation {
        let location = Location::new(self.location.0, self.location.1);
        let time_window = TimeWindow::new(Some(self.time_window.0), Some(self.time_window.1))
            .expect("Invalid time window");  // Handle error or use unwrap()
        
        Installation::new(
            self.idx,
            self.name,
            location,
            self.service_time, 
            time_window,
            self.deck_demand as u32,
            self.visit_frequency,
            self.inst_type,
            self.departure_spread,
        )
    }
}

// Custom deserializer for the location field
fn parse_location<'de, D>(deserializer: D) -> Result<(f64, f64), D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let s = s.trim_matches(|p| p == '[' || p == ']');
    let parts: Vec<&str> = s.split(',').collect();

    if parts.len() != 2 {
        return Err(serde::de::Error::custom("Invalid location format"));
    }

    let latitude = f64::from_str(parts[0].trim()).map_err(serde::de::Error::custom)?;
    let longitude = f64::from_str(parts[1].trim()).map_err(serde::de::Error::custom)?;
    
    Ok((latitude, longitude))
}

// Custom deserializer for the time_window field
fn parse_time_window<'de, D>(deserializer: D) -> Result<(u32, u32), D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let s = s.trim_matches(|p| p == '(' || p == ')');
    let parts: Vec<&str> = s.split(',').collect();

    if parts.len() != 2 {
        return Err(serde::de::Error::custom("Invalid time window format"));
    }

    let start = u32::from_str(parts[0].trim()).map_err(serde::de::Error::custom)?;
    let end = u32::from_str(parts[1].trim()).map_err(serde::de::Error::custom)?;
    
    Ok((start, end))
}

pub fn read_from_csv<T>(file_path: &str) -> Result<Vec<T>, Box<dyn Error>>
where
    T: DeserializeOwned,
{
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new()
        .delimiter(b',')
        .from_reader(file);

    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: T = result?;
        records.push(record);
    }

    Ok(records)
}

#[derive(Debug, Deserialize)]
pub struct BaseCSV {
    name: String,
    idx: u32,
    service_time: f64,
    #[serde(deserialize_with = "parse_time_window")]
    time_window: (u32, u32),
    longitude: f64,
    latitude: f64,
}

impl BaseCSV {
    // Method to convert BaseCSV to Base
    pub fn to_base(self) -> Base {
        let location = Location::new(self.latitude, self.longitude);
        let time_window = TimeWindow::new(Some(self.time_window.0), Some(self.time_window.1))
            .expect("Invalid time window");

        Base::new(
            self.idx,
            self.name,
            location,
            self.service_time,
            time_window,
        )

    }
}