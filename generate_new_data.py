import pandas as pd
import numpy as np

# Load original dataset
file_path = "data/MLE-Assignment.csv"
df = pd.read_csv(file_path)

# Select only required columns: hsi_id and features (0 to 447)
df = df.iloc[:, :449] 

# Generate new hsi_id values
new_hsi_ids = [f"imagoai_corn_{i}" for i in range(500, 550)]

# Generate new feature values based on existing data with slight variations
feature_columns = df.columns[1:]  # Exclude hsi_id
new_data = []

for hsi_id in new_hsi_ids:
    # Select a random existing row
    base_row = df.sample(n=1, random_state=np.random.randint(0, 1000)).iloc[:, 1:].values.flatten()
    
    # Apply slight variation (random noise)
    noise = np.random.normal(loc=0, scale=0.05, size=base_row.shape)  # Adjust noise scale if needed
    new_features = base_row + noise  # Add noise to the base row

    # Ensure values remain within realistic range
    new_features = np.clip(new_features, df.iloc[:, 1:].min().values, df.iloc[:, 1:].max().values)

    # Append new row
    new_data.append([hsi_id] + new_features.tolist())

# Create new DataFrame
new_df = pd.DataFrame(new_data, columns=["hsi_id"] + list(feature_columns))

# Save to CSV
output_file = "data/new_gen_data.csv"
new_df.to_csv(output_file, index=False)

print(f"New data saved to {output_file}")
