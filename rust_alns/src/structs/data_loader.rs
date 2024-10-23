use crate::structs::node::{Base, Installation};
use crate::structs::vessel::Vessel;
use crate::structs::csv_reader::{InstallationCSV, BaseCSV, read_from_csv};
use std::error::Error;

pub struct Data {
    pub installations: Vec<Installation>,
    pub vessels: Vec<Vessel>,
    pub base: Base,
}

// Load Installations
pub fn load_installations(file_path: &str) -> Result<Vec<Installation>, Box<dyn Error>> {
    let installation_csvs: Vec<InstallationCSV> = read_from_csv(file_path)?;

    let installations = installation_csvs
        .into_iter()
        .map(|record| record.to_installation())
        .collect();

    Ok(installations)
}

// Load Vessels
pub fn load_vessels(file_path: &str) -> Result<Vec<Vessel>, Box<dyn Error>> {
    let vessels: Vec<Vessel> = read_from_csv(file_path)?;
    Ok(vessels)
}

// Load Base
pub fn load_base(file_path: &str) -> Result<Base, Box<dyn Error>> {
    let base_csvs: Vec<BaseCSV> = read_from_csv(file_path)?;
    let base_csv = base_csvs.into_iter().next().ok_or("Empty CSV")?;

    // Use the to_base method to convert BaseCSV to Base
    Ok(base_csv.to_base())
}

// Load all data together
pub fn read_data(installations_path: &str, vessels_path: &str, base_path: &str) -> Result<Data, Box<dyn Error>> {
    let installations = load_installations(installations_path)?;
    let vessels = load_vessels(vessels_path)?;
    let base = load_base(base_path)?;

    Ok(Data {
        installations,
        vessels,
        base,
    })
}