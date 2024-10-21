use rust_alns::structs::csv_reader::read_installations_from_csv;  // Import the CSV reader function

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let installations = read_installations_from_csv("../sample/installations/SMALL_1/i_test1.csv")?;

    // Now you have a Vec<Installation> which you can use in your ALNS algorithm
    for installation in installations {
        println!("{:?}", installation);
    }

    Ok(())
}