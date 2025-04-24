use std::error::Error;
use std::fs::File;
use csv::ReaderBuilder;
use serde::{ser, Deserialize};
use crate::structs::node::{Installation, Location, Base};
use crate::structs::vessel::Vessel;
use crate::structs::time_window::TimeWindow;
use std::str::FromStr;
use serde::de::DeserializeOwned;

#[derive(Debug, Deserialize)]
pub struct InstallationCSV {
    #[serde(rename = "idx")]
    id: usize,
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
    time_window: (f64, f64),  // Changed from (u32, u32) to (f64, f64)
    service_time: f64,  // Changed to f64 to handle the decimal values
}

impl InstallationCSV {
    // Convert InstallationCSV to Installation object
    pub fn to_installation(self) -> Result<Installation, &'static str> {
        let location = Location::new(self.location.0, self.location.1);
        let time_window = TimeWindow::new(Some(self.time_window.0), Some(self.time_window.1))
            .expect("Invalid time window");  // Handle error or use unwrap()
        let installation = Installation::builder() 
            .id(self.id)
            .name(self.name)
            .location(location)
            .service_time(self.service_time)
            .time_window(time_window)
            .deck_demand(self.deck_demand as u32)
            .visit_frequency(self.visit_frequency)
            .installation_type(self.inst_type)
            .departure_spread(self.departure_spread)
            .build()?;
        Ok(installation)
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
fn parse_time_window<'de, D>(deserializer: D) -> Result<(f64, f64), D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    let s = s.trim_matches(|p| p == '(' || p == ')');
    let parts: Vec<&str> = s.split(',').collect();

    if parts.len() != 2 {
        return Err(serde::de::Error::custom("Invalid time window format"));
    }

    let start = f64::from_str(parts[0].trim()).map_err(serde::de::Error::custom)?;
    let end = f64::from_str(parts[1].trim()).map_err(serde::de::Error::custom)?;
    
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
    #[serde(rename = "idx")]
    id: usize,
    service_time: f64,
    #[serde(deserialize_with = "parse_time_window")]
    time_window: (f64, f64),  // Changed from (u32, u32) to (f64, f64)
    #[serde(deserialize_with = "parse_location")]
    location: (f64, f64),  // Parsed as tuple from "[latitude, longitude]"
}

impl BaseCSV {
    // Method to convert BaseCSV to Base
    pub fn to_base(self) -> Result<Base, &'static str> {
        let location = Location::new(self.location.0, self.location.1);
        let time_window = TimeWindow::new(Some(self.time_window.0), Some(self.time_window.1))
            .expect("Invalid time window");
        let base = Base::builder()
            .name(self.name)
            .id(self.id)
            .service_time(self.service_time)
            .location(location)
            .time_window(time_window)
            .build()?;
        Ok(base)
    }
}
#[derive(Debug, Deserialize)]
pub struct VesselCSV {
    #[serde(rename = "idx")]
    pub id: usize,
    pub name: String,
    pub deck_capacity: f64,
    pub bulk_capacity: f64,
    pub speed: f64,
    pub vessel_type: String,
    pub fcs: f64,
    pub fcw: f64,
    pub cost: f64,
}

impl VesselCSV {
    pub fn to_vessel(self) -> Result<Vessel, &'static str> {
        Vessel::builder()
            .id(self.id)
            .name(self.name)
            .deck_capacity(self.deck_capacity)
            .bulk_capacity(self.bulk_capacity)
            .speed(self.speed)
            .vessel_type(self.vessel_type)
            .fuel_consumption_sailing(self.fcs)
            .fuel_consumption_waiting(self.fcw)
            .cost(self.cost)
            .build()
    }
}