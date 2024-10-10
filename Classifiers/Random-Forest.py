import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the training and test data
train_rssi_diff_df = pd.read_excel('E:\\Latest\\Training-Data.xlsx', sheet_name=0)  # RSSI Difference
train_similarity_score_df = pd.read_excel('E:\\Latest\\Training-Data.xlsx', sheet_name=2)  # Similarity Score

test_rssi_diff_df = pd.read_excel('E:\\Latest\\Test-Data.xlsx', sheet_name=0)  # RSSI Difference
test_similarity_score_df = pd.read_excel('E:\\Latest\\Test-Data.xlsx', sheet_name=2)  # RSSI Similarity

# Merge the training data using both RSSI Difference and Similarity Score
train_df = pd.concat([train_rssi_diff_df[['RSSI Difference']], train_similarity_score_df[['RSSI Similarity', 'Similarity Score']]], axis=1).dropna()

# Merge the test data using both RSSI Difference and RSSI Similarity
test_df = pd.concat([test_rssi_diff_df[['RSSI Difference']], test_similarity_score_df[['RSSI Similarity']]], axis=1).dropna()

# Train the classifier using the Similarity Score to determine Sybil nodes
# Sybil nodes have a specific label in the Similarity Score (we will assume it's binary: 0 for non-Sybil, 1 for Sybil)
# Adjust this part based on your specific data if needed.
y_train = (train_df['Similarity Score'] > 0).astype(int)  # Consider values greater than 0 as Sybil nodes (1)

# Prepare the training data (RSSI Difference and RSSI Similarity as features)
X_train = train_df[['RSSI Difference', 'RSSI Similarity']]

# Prepare the test data
X_test = test_df[['RSSI Difference', 'RSSI Similarity']]

# Train a RandomForestClassifier model (binary classification)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict Sybil nodes for the test data
test_predictions_binary = classifier.predict(X_test)

# Assign 0 for non-Sybil nodes and 1 for Sybil nodes
test_df['Predicted Sybil Node'] = test_predictions_binary  # 0 = non-Sybil, 1 = Sybil

# Keep only the necessary columns (Pair, RSSI Similarity, Predicted Sybil Node)
output_df = test_similarity_score_df[['Pair', 'RSSI Similarity']]
output_df['Predicted Sybil Node'] = test_predictions_binary

# Save the results to a new Excel file
output_path = 'rf.xlsx'
output_df.to_excel(output_path, index=False)

print(f"Predicted Sybil node detection saved to {output_path}")
