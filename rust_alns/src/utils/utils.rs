pub fn intervals_overlap(start1: f64, end1: f64, start2: f64, end2: f64) -> bool {
    start1 < end2 && start2 < end1
}
//Tests:
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_intervals_overlap() {
        assert!(intervals_overlap(1.0, 3.0, 2.0, 4.0));
        assert!(!intervals_overlap(1.0, 2.0, 3.0, 4.0));
        assert!(intervals_overlap(1.0, 5.0, 2.0, 3.0));
        assert!(!intervals_overlap(1.0, 2.0, 2.0, 3.0));
    }
}