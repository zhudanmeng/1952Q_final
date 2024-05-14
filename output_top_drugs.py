import pandas as pd

# Load the latent space data for drugs and diseases
topics_drugs = pd.read_csv('final_A_SCMFDD', sep=" ", header=None)
topics_diseases = pd.read_csv('final_W_SCMFDD', sep=" ", header=None)

# Load the drug and disease names
drug_names = pd.read_csv('SCMFDD-L-drug.csv')
disease_names = pd.read_csv('SCMFDD-L-disease.csv', nrows=1323)

# Assign drug and disease names to the indices and columns
topics_drugs.index = drug_names['drug_id']
topics_diseases.columns = disease_names['name']

# Number of topics is the number of columns in topics_drugs (or topics_diseases)
num_topics = topics_drugs.shape[1]

# Output top 5 drugs and diseases for each topic
for topic in range(num_topics):
    # Get top 5 drugs for the current topic
    # top_drugs = topics_drugs.nlargest(5, topic).index.tolist()
    # print(f"Topic {topic + 1} - Top 5 Drugs: {', '.join(top_drugs)}")

    # Get top 5 diseases for the current topic
    top_diseases = topics_diseases.T.nlargest(5, topic).index.tolist()
    top_diseases = str(top_diseases)
    print(top_diseases)
    # print(f"Topic {topic + 1} - Top 5 Diseases: {', '.join(top_diseases)}")
