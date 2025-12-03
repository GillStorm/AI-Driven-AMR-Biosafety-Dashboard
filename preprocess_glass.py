import pandas as pd

def preprocess_glass_data(input_file, output_file):
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # Column mapping
    # "Iso3","CountryTerritoryArea","WHORegionName","Year","Specimen","PathogenName","AbTargets","TotalSpecimenIsolates","InterpretableAST","Resistant","PercentResistant"
    # -> location,year,pathogen,antibiotic,n_tested,n_resistant
    
    rename_map = {
        "CountryTerritoryArea": "location",
        "Year": "year",
        "PathogenName": "pathogen",
        "AbTargets": "antibiotic",
        "InterpretableAST": "n_tested",
        "Resistant": "n_resistant"
    }
    
    # Check if columns exist
    missing_cols = [c for c in rename_map.keys() if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in input file: {missing_cols}")
        return

    df = df.rename(columns=rename_map)
    
    # Select only required columns
    required_cols = ["location", "year", "pathogen", "antibiotic", "n_tested", "n_resistant"]
    df = df[required_cols]

    # Clean pathogen names
    # Map common names to short forms used in the dashboard if needed, or just keep them.
    # The dashboard doesn't seem to strictly enforce pathogen names, but shorter is better for plots.
    pathogen_map = {
        "Escherichia coli": "E.coli",
        "Klebsiella pneumoniae": "K.pneumoniae",
        "Staphylococcus aureus": "S.aureus",
        "Salmonella sp.": "Salmonella",
        "Streptococcus pneumoniae": "S.pneumoniae",
        "Neisseria gonorrhoeae": "N.gonorrhoeae",
        "Acinetobacter spp.": "Acinetobacter",
        "Pseudomonas aeruginosa": "P.aeruginosa"
    }
    df["pathogen"] = df["pathogen"].replace(pathogen_map)

    # Clean antibiotic names (lowercase)
    df["antibiotic"] = df["antibiotic"].str.lower()

    # Drop rows with missing values in critical columns
    df = df.dropna(subset=["location", "year", "pathogen", "antibiotic", "n_tested", "n_resistant"])

    # Ensure numeric types
    df["n_tested"] = pd.to_numeric(df["n_tested"], errors='coerce')
    df["n_resistant"] = pd.to_numeric(df["n_resistant"], errors='coerce')
    df = df.dropna(subset=["n_tested", "n_resistant"])

    print(f"Processed {len(df)} rows.")
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    preprocess_glass_data("who_glass_2022.csv", "who_glass_2022_processed.csv")
