import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
import seaborn as sns


mcdonalds = pd.read_csv('mcdonalds.csv')

print(mcdonalds.columns.tolist())

print(mcdonalds.shape)

print(mcdonalds.head(3))

MD_x = mcdonalds.iloc[:, 0:11]

MD_x_matrix = (MD_x == "Yes").astype(int)

column_means = MD_x_matrix.mean().round(2)

print(column_means)

pca = PCA()
MD_pca = pca.fit(MD_x_matrix)
MD_p=pca.fit(MD_x_matrix)

SD=np.sqrt(pca.explained_variance_)
PV=pca.explained_variance_ratio_
index=[]
for i in range(len(SD)):
    i=i+1
    index.append("PC{}".format(i))

sum=pd.DataFrame({
    "Standard deviation":SD,"Proportion of Variance":PV,"Cumulative Proportion":PV.cumsum()
},index=index)
sum

print("Standard Deviation:\n",SD.round(1))

rotation_matrix = MD_pca.components_.T

columns = [f"PC{i+1}" for i in range(rotation_matrix.shape[1])]
index = MD_x.columns
rotation_df = pd.DataFrame(data=np.round(rotation_matrix, 3), index=index, columns=columns)

print("Rotation (n x k) = (11 x 11):")
print(-rotation_df)

loadings=-pca.components_.T
plt.figure(figsize=(8,5))
# Plot the predicted PCA components

plt.scatter(MD_pca_components_[:, 0], MD_pca_components_[:, 1], color='grey')

plt.xlim(-0.6, 0.9)
plt.ylim(-0.9, 0.9)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of McDonalds Data')
plt.grid(True)

for i in range(loadings.shape[0]):
  plt.arrow(0,0,loadings[i,0],loadings[i,1],color='r',alpha=0.5,head_width=0.05, head_length=0.05, linewidth=1.5)
  plt.text(loadings[i,0],loadings[i,1],MD_x.columns[i],color='g',ha='center',va='center',)

plt.show()


np.random.seed(1234)

nrep=10

num_segments = range(1, 9)
MD_km28={}

within_cluster_distances = []
for k in num_segments:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=1234)
    kmeans.fit(MD_x_matrix)
    within_cluster_distances.append(kmeans.inertia_)
    MD_km28[str(k)] = kmeans

plt.bar(num_segments, within_cluster_distances)
plt.xlabel('Number of Segments')
plt.ylabel('Sum of within cluster distances')
plt.title('Segmentation Results')
plt.show()

from sklearn.metrics import adjusted_rand_score
np.random.seed(1234)
nboot = 100
nrep = 10

bootstrap_samples = []
for _ in range(nboot):
    bootstrap_sample = resample(MD_x_matrix.values, random_state=1234)
    bootstrap_samples.append(bootstrap_sample)

adjusted_rand_index = []
num_segments = range(2, 9)
for k in num_segments:
    stability_scores = []
    for bootstrap_sample in bootstrap_samples:
        kmeans = KMeans(n_clusters=k, n_init=nrep, random_state=1234)
        kmeans.fit(bootstrap_sample)
        cluster_labels = kmeans.predict(bootstrap_sample)
        true_labels = kmeans.predict(MD_x_matrix.values)
        stability_score = adjusted_rand_score(true_labels, cluster_labels)
        stability_scores.append(stability_score)
    adjusted_rand_index.append(stability_scores)

adjusted_rand_index = np.array(adjusted_rand_index).T

plt.boxplot(adjusted_rand_index, labels=num_segments, whis=10)
plt.xlabel("Number of segments")
plt.ylabel("Adjusted Rand Index")
plt.title("Bootstrap Flexclust")
plt.show()

range_values = (0, 1)
num_bins = 10
max_frequency = 200

fig, axs = plt.subplots(2, 2, figsize=(12, 8))


for i in range(1, 5):
    labels = MD_km28[str(i)].predict(MD_x_matrix)
    similarities = MD_km28[str(i)].transform(MD_x_matrix).min(axis=1)
    row = (i - 1) // 2
    col = (i - 1) % 2

    axs[row, col].hist(similarities, bins=num_bins, range=range_values)
    axs[row, col].set_xlabel('Similarity')
    axs[row, col].set_ylabel('Frequency')
    axs[row, col].set_title('cluster {}'.format(i))

    axs[row, col].set_xlim(range_values)
    axs[row, col].set_ylim(0, max_frequency)


    axs[row, col].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

plt.tight_layout()
plt.show()


num_segments = range(2, 9)
segment_stability = []

for segment in num_segments:
    labels_segment = MD_km28[str(segment)].predict(MD_x_matrix)
    segment_stability.append(labels_segment)

plt.figure(figsize=(6, 6))

for i, segment in enumerate(num_segments):
    stability = [np.mean(segment_stability[i] == labels) for labels in segment_stability]
    plt.plot(num_segments, stability, marker='o', label=f'Segment {segment}')

plt.xlabel('Number of Segments')
plt.ylabel('Segment Level Stability')
plt.title('Segment Level Stability Across Solutions (SLSA) Plot')
plt.xticks(num_segments)
plt.legend()
plt.grid(True)
plt.show()


segment_solutions = ["2", "3", "4", "5"]


segment_labels = {}
segment_similarities = {}

for segment in segment_solutions:
    model = MD_km28[segment]
    labels = model.predict(MD_x_matrix)
    similarities = model.transform(MD_x_matrix).min(axis=1)
    
    segment_labels[segment] = labels
    segment_similarities[segment] = similarities


segment_stability_values = [
    similarities / np.max(similarities) for similarities in segment_similarities.values()
]

plt.boxplot(segment_stability_values, whis=1.5)
plt.xlabel("Segment Number")
plt.ylabel("Segment Stability")
plt.xticks(range(1, len(segment_solutions) + 1), segment_solutions)
plt.ylim(0, 1)
plt.title("Segment Level Stability within Solutions")

plt.show()


from scipy.stats import entropy

np.random.seed(1234)
k_values = range(2, 9)

results = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=1234)
    model.fit(MD_x_matrix.values)
    
    iter_val = model.n_iter_
    converged = True
    log_likelihood = -model.inertia_
    n_samples = MD_x_matrix.shape[0]
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + np.log(n_samples) * k
    
    labels = model.labels_
    counts = np.bincount(labels)
    probs = counts / float(counts.sum())
    class_entropy = entropy(probs)
    icl = bic - class_entropy
    
    results.append((iter_val, converged, k, k, log_likelihood, aic, bic, icl))

columns = ['iter', 'converged', 'k', 'k0', 'logLik', 'AIC', 'BIC', 'ICL']
MD_m28 = pd.DataFrame(results, columns=columns)

print(MD_m28)


from scipy.stats import entropy

np.random.seed(1234)
MD = np.random.rand(100, 5)

k_values = range(2, 9)
results = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=1234)
    model.fit(MD_x_matrix)
    labels = model.labels_
    log_likelihood = -model.inertia_
    n_samples = MD_x_matrix.shape[0]
    aic = -2 * log_likelihood + 2 * k
    bic = -2 * log_likelihood + np.log(n_samples) * k
    counts = np.bincount(labels)
    probs = counts / float(counts.sum())
    class_entropy = entropy(probs)
    icl = bic - class_entropy
    
    results.append((k, log_likelihood, aic, bic, icl))

results_df = pd.DataFrame(results, columns=['k', 'logLik', 'AIC', 'BIC', 'ICL'])

kmeans_model = KMeans(n_clusters=k, random_state=1234).fit(MD)
mixture_model = KMeans(n_clusters=k, random_state=1234).fit(MD)

plt.figure(figsize=(10, 6))
plt.plot(results_df['k'], results_df['AIC'], label='AIC', marker='o')
plt.plot(results_df['k'], results_df['BIC'], label='BIC', marker='o')
plt.plot(results_df['k'], results_df['ICL'], label='ICL', marker='o')

plt.xlabel('Number of Components')
plt.ylabel('Value of Information Criteria')
plt.title('Information Criteria vs. Number of Components')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.mixture import GaussianMixture

np.random.seed(1234)
MD_x_matrix = np.random.rand(100, 5)  

k = 4
kmeans_model = KMeans(n_clusters=k, random_state=1234).fit(MD_x_matrix)
kmeans_labels = kmeans_model.labels_

gmm_model = GaussianMixture(n_components=k, covariance_type='full', random_state=1234)
gmm_labels = gmm_model.fit_predict(MD_x_matrix)

contingency_table = pd.crosstab(kmeans_labels, gmm_labels,rownames=['kmeans'], colnames=['mixture'])
print(contingency_table)


from sklearn.mixture import GaussianMixture
import numpy as np

gmm_m4a = GaussianMixture(n_components=4)
gmm_m4a.fit(MD_x_matrix)

log_likelihood_m4a = gmm_m4a.score(MD_x_matrix)

gmm_m4 = GaussianMixture(n_components=4)
gmm_m4.fit(MD_x_matrix)

log_likelihood_m4 = gmm_m4.score(MD_x_matrix)

print("Log-likelihood for MD.m4a:", log_likelihood_m4a)
print("Log-likelihood for MD.m4:", log_likelihood_m4)

import pandas as pd

data = {
    'Like': [
        'I hate it!-5', '-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4', 'I love it!+5'
    ],
    'Frequency': [152, 71, 73, 59, 58, 169, 152, 187, 229, 160, 143]
}

df = pd.DataFrame(data)

ordinal_to_numeric = {
    'I hate it!-5': -5,
    '-4': -4,
    '-3': -3,
    '-2': -2,
    '-1': -1,
    '0': 0,
    '+1': 1,
    '+2': 2,
    '+3': 3,
    '+4': 4,
    'I love it!+5': 5
}

df['Like.n'] = df['Like'].apply(lambda x: 6 - ordinal_to_numeric[x])

print("Data with numeric conversion:")
print(df)

