use serde::Deserialize;
use std::error::Error;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use rand::seq::SliceRandom;
use rand::Rng;
use rand::thread_rng;
use crate::structs::time_window::TimeWindow;

#[derive(Debug, Clone, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
}

impl Default for Location {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
        }
    }
}

impl Location {
    pub fn new(latitude: f64, longitude: f64) -> Self {
        Self { latitude, longitude }
    }
}

pub trait HasLocation {
    fn get_location(&self) -> &Location;
}
pub trait HasTimeWindows {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow>;
    fn get_tw(&self) -> (Option<f64>, Option<f64>);
}

// New composite trait that combines HasLocation and HasTimeWindows
pub trait HasLocationAndTW: HasLocation + HasTimeWindows {}

// Blanket implementation for any type that implements both required traits
impl<T: HasLocation + HasTimeWindows> HasLocationAndTW for T {}

#[derive(Debug, Clone, Deserialize)]
pub struct Node {
    pub name: String,
    pub location: Location,
}

impl Node {
    pub fn new(name: String, location: Location) -> Self {
        Self { name, location }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Installation {
    pub id: usize,
    pub node: Node,
    pub deck_demand: u32,
    pub visit_frequency: u32,
    pub installation_type: String,
    pub departure_spread: u32,
    pub service_time: f64,
    pub time_window: TimeWindow,
    pub service_tw: Vec<TimeWindow>,
}

impl Installation {
    pub fn builder() -> InstallationBuilder {
        InstallationBuilder::default()
    }

    pub fn generate_visit_scenario(&self) -> Vec<u32> {
        // TODO: Implement visit scenario generation
        // Placeholder implementation: random visit days
        let mut rng = thread_rng();
        (0..DAYS_IN_PERIOD)
            .map(|_| if rng.gen_range(0..100) < self.visit_frequency { 1 } else { 0 })
            .collect()
    }
    
    pub fn get_service_time(&self) -> f64 {
        self.service_time
    }
}
#[derive(Default)]
pub struct InstallationBuilder {
    id: usize,
    name: String,
    location: Location,
    deck_demand: u32,
    visit_frequency: u32,
    installation_type: String,
    departure_spread: u32,
    service_time: f64,
    time_window: TimeWindow,
    service_tw: Vec<TimeWindow>,
}

impl InstallationBuilder {
    // Private helper method to generate service time windows
    fn create_service_time_windows(&self, time_window: &TimeWindow) -> Vec<TimeWindow> {
        // If time_window is (0, 24) or default (None, None), use default time windows that always return true
        if (time_window.earliest == Some(0.0) && time_window.latest == Some(24.0)) || 
           (time_window.earliest.is_none() && time_window.latest.is_none()) {
            vec![TimeWindow::default(); DAYS_IN_PERIOD as usize]
        } else {
            (0..DAYS_IN_PERIOD)
                .map(|day| TimeWindow::new(
                    time_window.earliest.map(|earliest| (earliest + day as f64 * HOURS_IN_DAY as f64)),
                    time_window.latest.map(|latest| (latest + day as f64 * HOURS_IN_DAY as f64) - self.service_time),
                ).unwrap())
                .collect()
        }
    }

    pub fn id(mut self, id: usize) -> Self {
        self.id = id;
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn location(mut self, location: Location) -> Self {
        self.location = location;
        self
    }

    pub fn deck_demand(mut self, deck_demand: u32) -> Self {
        self.deck_demand = deck_demand;
        self
    }

    pub fn visit_frequency(mut self, visit_frequency: u32) -> Self {
        self.visit_frequency = visit_frequency;
        self
    }

    pub fn installation_type(mut self, installation_type: String) -> Self {
        self.installation_type = installation_type;
        self
    }

    pub fn departure_spread(mut self, departure_spread: u32) -> Self {
        self.departure_spread = departure_spread;
        self
    }

    pub fn service_time(mut self, service_time: f64) -> Self {
        self.service_time = service_time;
        self
    }

    pub fn time_window(mut self, time_window: TimeWindow) -> Self {
        self.time_window = time_window;
        self
    }

    // Creates a time window for each day in the period by adding 24-hour shifts
    pub fn service_time_windows(mut self, time_window: TimeWindow) -> Self {
        self.service_tw = self.create_service_time_windows(&time_window);
        self
    }

    pub fn build(self) -> Result<Installation, &'static str> {
        // Initialize service_TW from time_window if not explicitly set
        let service_tw = if self.service_tw.is_empty() {
            self.create_service_time_windows(&self.time_window)
        } else {
            self.service_tw
        };
        
        Ok(Installation {
            id: self.id,
            node: Node::new(self.name, self.location),
            deck_demand: self.deck_demand,
            visit_frequency: self.visit_frequency,
            installation_type: self.installation_type,
            departure_spread: self.departure_spread,
            service_time: self.service_time,
            time_window: self.time_window,
            service_tw,
        })
    }
}

impl HasLocation for Installation {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Installation {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow> {
        &self.service_tw
    }
    fn get_tw(&self) -> (Option<f64>, Option<f64>) {
        let tw = self.get_service_time_windows();
        (tw[0].earliest, tw[0].latest)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Base {
    pub id: usize,
    pub node: Node,
    pub service_time: f64,
    pub time_window: TimeWindow,
    pub service_tw: Vec<TimeWindow>,
}

impl Base {
    pub fn builder() -> BaseBuilder {
        BaseBuilder::default()
    }
}

#[derive(Default)]
pub struct BaseBuilder {
    id: Option<usize>,
    name: Option<String>,
    location: Option<Location>,
    service_time: Option<f64>,
    time_window: Option<TimeWindow>,
    service_tw: Vec<TimeWindow>,
}

impl BaseBuilder {
    pub fn id(mut self, id: usize) -> Self {
        self.id = Some(id);
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn location(mut self, location: Location) -> Self {
        self.location = Some(location);
        self
    }

    pub fn service_time(mut self, service_time: f64) -> Self {
        self.service_time = Some(service_time);
        self
    }

    pub fn time_window(mut self, time_window: TimeWindow) -> Self {
        self.time_window = Some(time_window);
        self
    }

    // Creates a time window for each day in the period by adding 24-hour shifts
    pub fn service_time_windows(mut self, time_window: TimeWindow) -> Self {
        // If time_window is (0, 24) or default (None, None), use default time windows that always return true
        if (time_window.earliest == Some(0.0) && time_window.latest == Some(24.0)) || 
           (time_window.earliest.is_none() && time_window.latest.is_none()) {
            self.service_tw = vec![TimeWindow::default(); DAYS_IN_PERIOD as usize];
        } else {
            self.service_tw = (0..DAYS_IN_PERIOD)
                .map(|day| TimeWindow::new(
                    time_window.earliest.map(|earliest| (earliest + day as f64 * HOURS_IN_DAY as f64) % HOURS_IN_PERIOD as f64),
                    time_window.latest.map(|latest| (latest + day as f64 * HOURS_IN_DAY as f64 - self.service_time.unwrap_or(0.0)) % HOURS_IN_PERIOD as f64),
                ).unwrap())
                .collect()
        }
        self
    }

    pub fn build(self) -> Result<Base, &'static str> {
        let time_window = self.time_window.ok_or("time_window is required")?;
        
        // If service_tw wasn't explicitly set, generate it from the time_window
        let service_tw = if self.service_tw.is_empty() {
            if (time_window.earliest == Some(0.0) && time_window.latest == Some(24.0)) || 
            (time_window.earliest.is_none() && time_window.latest.is_none()) {
                vec![TimeWindow::default(); DAYS_IN_PERIOD as usize]
            } else {
                (0..DAYS_IN_PERIOD)
                    .map(|day| TimeWindow::new(
                        time_window.earliest.map(|earliest| (earliest + day as f64 * HOURS_IN_DAY as f64) % HOURS_IN_PERIOD as f64),
                        time_window.latest.map(|latest| (latest + day as f64 * HOURS_IN_DAY as f64 - self.service_time.unwrap_or(0.0)) % HOURS_IN_PERIOD as f64),
                    ).unwrap())
                    .collect()
            }
        } else {
            self.service_tw
        };
        
        Ok(Base {
            id: self.id.ok_or("id is required")?,
            node: Node::new(
                self.name.ok_or("name is required")?,
                self.location.ok_or("location is required")?,
            ),
            service_time: self.service_time.ok_or("service_time is required")?,
            time_window,
            service_tw,
        })
    }
}

impl HasLocation for Base {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Base {
    fn get_service_time_windows(&self) -> &Vec<TimeWindow> {
        &self.service_tw
    }
    fn get_tw(&self) -> (Option<f64>, Option<f64>) {
        let tw = self.get_service_time_windows();
        (tw[0].earliest, tw[0].latest)
    }
}
