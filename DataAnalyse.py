import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('phishing_email.csv')

# print(df.info())

print("Ilościowo:\n", df['EmailLabel'].value_counts())

df['EmailLabel'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Rozkład klas (Phishing vs Legit)')
plt.xlabel('Klasa')
plt.ylabel('Liczba emaili')
plt.xticks(rotation=0)
plt.show()

print('Empty mails: ', df['EmailText'].isna().sum())
print('Empty labels: ', df['EmailLabel'].isna().sum())

df['EmailText'] = df['EmailText'].str.strip()
print('Duplikaty maili: ', df['EmailText'].duplicated().sum())

print("Max długość maila:", df['EmailText'].str.len().max())
print("Średnia długość maila:", df['EmailText'].str.len().mean())

long_mails = df['EmailText'].str.len() > 512
print("Liczba maili > 512 znaków:", long_mails.sum())

over_limit = df[df['EmailText'].str.len() > 512]['EmailText'].str.len() - 512

print("Średnio o ile przekraczają 512:", over_limit.mean())
print("Maksymalne przekroczenie:", over_limit.max())
print("Minimalne przekroczenie:", over_limit.min())


def get_top_words(text_series, n=20):
    words = " ".join(text_series.astype(str)).lower().split()
    return Counter(words).most_common(n)


print("Top słowa w Phishingu:", get_top_words(df[df['EmailLabel'] == 1]['EmailText']))
print("Top słowa w Legit:", get_top_words(df[df['EmailLabel'] == 0]['EmailText']))

vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['EmailText'])

batch_size = 2000
num_rows = tfidf_matrix.shape[0]
indices_to_drop = set()
displayed_count = 0
max_examples = 80000

for i in range(0, num_rows, batch_size):
    end_i = min(i + batch_size, num_rows)
    chunk = tfidf_matrix[i:end_i]

    sim_matrix = cosine_similarity(chunk, tfidf_matrix)

    rows, cols = np.where((sim_matrix > 0.9) & (sim_matrix < 0.99))

    for r, c in zip(rows, cols):
        actual_r = r + i
        if actual_r < c:
            indices_to_drop.add(c)
            displayed_count += 1
            similarity_score = sim_matrix[r, c]

            # print(f"\nPara nr {displayed_count} | Podobieństwo: {similarity_score:.4f}")
            # print(f"Indeks {actual_r} (Zostaje): {df.iloc[actual_r]['EmailText']}")
            # print(f"Indeks {c} (Do usunięcia): {df.iloc[c]['EmailText']}")
            # print("-" * 50)
            if displayed_count >= max_examples:
                break
    if displayed_count >= max_examples:
        break

print(f"Gotowe. Masz {len(indices_to_drop)} indeksów do wywalenia.")

df.drop(index=df.index[list(indices_to_drop)], inplace=True)
df.drop_duplicates(subset=['EmailText'], inplace=True)
df.dropna(subset=['EmailText', 'EmailLabel'], inplace=True)

print("\n--- STAN KOŃCOWY ---")
print("Ilościowo po wszystkich zabiegach:")
print(df['EmailLabel'].value_counts())

print("Top słowa w Legit: po czyszceniu", get_top_words(df[df['EmailLabel'] == 0]['EmailText']))
print("Top słowa w Phishing: po czyszceniu", get_top_words(df[df['EmailLabel'] == 1]['EmailText']))

df['EmailLabel'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Rozkład klas po czyszczeniu i deduplikacji')
plt.show()