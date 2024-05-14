import pandas as pd
import matplotlib.pyplot as plt

# Load the latent space data
topics_drugs = pd.read_csv('final_A_SCMFDD_S', sep=" ", header=None)
topics_diseases = pd.read_csv('final_W_SCMFDD_S', sep=" ", header=None)



# Load the drug names
drug_names = pd.read_csv('drug.csv')
topics_drugs.index = drug_names['drug_id']

disease_names = pd.read_csv('disease.csv', nrows=266)
topics_diseases.columns = disease_names['name']

# Number of topics
num_topics = topics_drugs.shape[1]

# Plotting the top n drugs for each topic
for topic in range(2):
    plt.figure(figsize=(10, 8))
    # Get top n drugs for the topic
    top_drugs = topics_drugs.nlargest(5, topic)
    # Create a bar plot
    plt.barh(top_drugs.index, top_drugs[topic])
    plt.xlabel('Score')
    plt.ylabel('Drug Name')
    plt.title(f'Top 5 Drugs for Topic {topic + 1}')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest value on top
    plt.show()

# Plotting the top n diseases for each topic
for topic in range(2):
    plt.figure(figsize=(10, 8))
    # Get top n diseases for the topic
    top_diseases = topics_diseases.T.nlargest(5, topic)
    # Create a bar plot
    plt.barh(top_diseases.index, top_diseases[topic])
    plt.xlabel('Score')
    plt.ylabel('Disease Name')
    plt.title(f'Top 5 diseases for Topic {topic + 1}')
    plt.gca().invert_yaxis()  # Invert the y-axis to have the highest value on top
    plt.show()
