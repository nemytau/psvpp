use crate::structs::constants::{
    DAYS_IN_PERIOD, HOURS_IN_DAY, HOURS_IN_PERIOD, REL_DEPARTURE_TIME,
};
use crate::structs::time_window::TimeWindow;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::error::Error;

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
        Self {
            latitude,
            longitude,
        }
    }
}

pub trait HasLocation {
    fn get_location(&self) -> &Location;
}
pub trait HasTimeWindows {
    fn get_service_time_windows(&self) -> &TimeWindow;
    fn earliest_departure_after_service(&self, arrival_time: f64) -> f64;
    fn get_tw(&self) -> (Option<f64>, Option<f64>) {
        let tw = self.get_service_time_windows();
        (tw.earliest, tw.latest)
    }
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
    pub service_time_window: TimeWindow,
}

impl Installation {
    pub fn builder() -> InstallationBuilder {
        InstallationBuilder::default()
    }

    // Generates a departure scenario for the installation. Returns a vector of days.
    // The days are represented as indices in the range [0, DAYS_IN_PERIOD).
    pub fn generate_departure_scenario(&self, rng: &mut impl Rng) -> Vec<usize> {
        let visit_count = self.visit_frequency as usize;
        let departure_spread = self.departure_spread as i32;
        assert!(visit_count > 0);
        assert!(DAYS_IN_PERIOD >= visit_count as u32);

        let mut scenario = BTreeSet::new();
        let mut attempts = 0;
        const MAX_TRIES: usize = 100;

        while scenario.len() < visit_count && attempts < MAX_TRIES {
            let day = rng.gen_range(0..DAYS_IN_PERIOD);
            let valid = scenario.iter().all(|&d| {
                let diff = (d as i32 - day as i32).abs();
                let wrapped = DAYS_IN_PERIOD as i32 - diff;
                diff.min(wrapped) >= departure_spread
            });

            if valid {
                scenario.insert(day as usize);
            }
            attempts += 1;
        }

        if scenario.len() < visit_count {
            panic!(
                "Could not generate a departure scenario for installation {} ({}) with frequency {} and departure spread {}",
                self.id,
                self.node.name,
                visit_count,
                departure_spread
            );
        }

        scenario.into_iter().collect()
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
    service_time_window: Option<TimeWindow>,
}

impl InstallationBuilder {
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

    pub fn service_tw(mut self, time_window: TimeWindow) -> Self {
        self.service_time_window = Some(time_window);
        self
    }

    pub fn build(self) -> Result<Installation, &'static str> {
        // Calculate service time window if not explicitly set
        let service_time_window = self.service_time_window.unwrap_or_else(|| {
            let tw = &self.time_window;
            let service_time = self.service_time;
            if let (Some(start), Some(end)) = (tw.earliest, tw.latest) {
                if (end - start).abs() >= 24.0 {
                    TimeWindow::new(Some(start), Some(end)).unwrap_or_default()
                } else {
                    TimeWindow::new(Some(start), Some(end - service_time)).unwrap_or_default()
                }
            } else {
                TimeWindow::new(tw.earliest, tw.latest).unwrap_or_default()
            }
        });
        Ok(Installation {
            id: self.id,
            node: Node::new(self.name, self.location),
            deck_demand: self.deck_demand,
            visit_frequency: self.visit_frequency,
            installation_type: self.installation_type,
            departure_spread: self.departure_spread,
            service_time: self.service_time,
            time_window: self.time_window,
            service_time_window,
        })
    }
}

impl HasLocation for Installation {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Installation {
    fn get_service_time_windows(&self) -> &TimeWindow {
        &self.service_time_window
    }
    fn earliest_departure_after_service(&self, arrival_time: f64) -> f64 {
        let service_tw = &self.service_time_window;
        if service_tw.is_open {
            return arrival_time + self.service_time;
        }
        let relative_arrival_time = arrival_time % HOURS_IN_DAY as f64;
        if service_tw.contains(relative_arrival_time) {
            return arrival_time + self.service_time;
        }
        if relative_arrival_time < service_tw.earliest.unwrap() {
            return service_tw.earliest.unwrap() + self.service_time;
        } else {
            return HOURS_IN_DAY as f64 - service_tw.latest.unwrap()
                + service_tw.earliest.unwrap()
                + self.service_time;
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Base {
    pub id: usize,
    pub node: Node,
    pub service_time: f64,
    pub time_window: TimeWindow,
    pub service_time_window: TimeWindow,
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
    service_time_window: Option<TimeWindow>,
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

    pub fn service_tw(mut self, time_window: TimeWindow) -> Self {
        self.service_time_window = Some(time_window);
        self
    }

    pub fn build(self) -> Result<Base, &'static str> {
        let time_window = self.time_window.ok_or("time_window is required")?;
        let service_time = self.service_time.ok_or("service_time is required")?;

        // Calculate service time window if not explicitly set
        let service_time_window = self.service_time_window.unwrap_or_else(|| {
            if let (Some(start), Some(end)) = (time_window.earliest, time_window.latest) {
                if (end - start).abs() >= 24.0 {
                    TimeWindow::new(Some(start), Some(end)).unwrap_or_default()
                } else {
                    TimeWindow::new(Some(start), Some(end - service_time)).unwrap_or_default()
                }
            } else {
                TimeWindow::new(time_window.earliest, time_window.latest).unwrap_or_default()
            }
        });

        Ok(Base {
            id: self.id.ok_or("id is required")?,
            node: Node::new(
                self.name.ok_or("name is required")?,
                self.location.ok_or("location is required")?,
            ),
            service_time,
            time_window,
            service_time_window,
        })
    }
}

impl HasLocation for Base {
    fn get_location(&self) -> &Location {
        &self.node.location
    }
}

impl HasTimeWindows for Base {
    fn get_service_time_windows(&self) -> &TimeWindow {
        &self.service_time_window
    }
    fn earliest_departure_after_service(&self, arrival_time: f64) -> f64 {
        let service_tw = &self.service_time_window;
        if service_tw.is_open {
            return arrival_time + self.service_time;
        }
        let relative_arrival_time = arrival_time % HOURS_IN_DAY as f64;
        if service_tw.contains(relative_arrival_time) {
            return arrival_time + self.service_time;
        }
        if relative_arrival_time < service_tw.earliest.unwrap() {
            return service_tw.earliest.unwrap() + self.service_time;
        } else {
            return HOURS_IN_DAY as f64 - service_tw.latest.unwrap()
                + service_tw.earliest.unwrap()
                + self.service_time;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structs::constants::DAYS_IN_PERIOD;
    use rand::{rngs::StdRng, SeedableRng};

    fn assert_valid_scenario(scenario: &[usize], freq: usize, spread: i32) {
        assert_eq!(scenario.len(), freq, "Wrong number of visit days");
        for (i, &day1) in scenario.iter().enumerate() {
            for &day2 in scenario.iter().skip(i + 1) {
                let diff = (day1 as i32 - day2 as i32).abs();
                let wrapped = DAYS_IN_PERIOD as i32 - diff;
                assert!(
                    diff.min(wrapped) >= spread,
                    "Spread condition violated: day1={}, day2={}, spread={}",
                    day1,
                    day2,
                    spread
                );
            }
        }
    }

    #[test]
    fn test_departure_scenario_normal_case() {
        let mut rng = StdRng::seed_from_u64(42);
        let freq = 3;
        let spread = 2;
        let installation = Installation::builder()
            .id(1)
            .name("Test".into())
            .location(Location::new(0.0, 0.0))
            .visit_frequency(freq)
            .departure_spread(spread)
            .build()
            .unwrap();

        let scenario = installation.generate_departure_scenario(&mut rng);
        assert_valid_scenario(&scenario, freq as usize, spread as i32);
    }

    #[test]
    fn test_departure_scenario_minimal_case() {
        let mut rng = StdRng::seed_from_u64(11);
        let freq = 1;
        let spread = 0;
        let installation = Installation::builder()
            .id(2)
            .name("OneVisit".into())
            .location(Location::new(0.0, 0.0))
            .visit_frequency(freq)
            .departure_spread(spread)
            .build()
            .unwrap();

        let scenario = installation.generate_departure_scenario(&mut rng);
        assert_valid_scenario(&scenario, freq as usize, spread as i32);
    }

    #[test]
    #[should_panic(expected = "Could not generate a departure scenario")]
    fn test_departure_scenario_too_tight_to_fit() {
        let mut rng = StdRng::seed_from_u64(99);
        let freq = 4;
        let spread = 3; // 4 * 3 = 12 > DAYS_IN_PERIOD, so not all can be placed
        let installation = Installation::builder()
            .id(3)
            .name("TooTight".into())
            .location(Location::new(0.0, 0.0))
            .visit_frequency(freq)
            .departure_spread(spread)
            .build()
            .unwrap();

        let scenario = installation.generate_departure_scenario(&mut rng);
        assert!(
            scenario.len() < freq as usize,
            "Scenario should fail to place all days with too tight spread"
        );
    }

    #[test]
    fn test_departure_scenario_determinism() {
        let freq = 3;
        let spread = 1;
        let installation = Installation::builder()
            .id(4)
            .name("Repeatable".into())
            .location(Location::new(0.0, 0.0))
            .visit_frequency(freq)
            .departure_spread(spread)
            .build()
            .unwrap();

        let mut rng1 = StdRng::seed_from_u64(123);
        let mut rng2 = StdRng::seed_from_u64(123);

        let s1 = installation.generate_departure_scenario(&mut rng1);
        let s2 = installation.generate_departure_scenario(&mut rng2);
        assert_eq!(s1, s2, "Same seed should produce same scenario");
    }
}
