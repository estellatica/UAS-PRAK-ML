# === SPAM CLASSIFIER with GRAPH & LOG FILE ===

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Output file
    with open("hasil_output.txt", "w", encoding="utf-8") as f:

        f.write("=== KLASIFIKASI SPAM SMS ===\n\n")

        # 1. Load dataset
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep='\t', header=None, names=['label', 'text'])
        f.write(f"Jumlah data: {len(df)}\n")

        # 2. Preprocessing label
        df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

        # 3. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['label_num'], test_size=0.2, random_state=42)

        # 4. Vectorizer
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # 5. Naive Bayes
        nb = MultinomialNB()
        nb.fit(X_train_vec, y_train)
        pred_nb = nb.predict(X_test_vec)
        acc_nb = accuracy_score(y_test, pred_nb)

        # 6. SVM
        svm = SVC()
        svm.fit(X_train_vec, y_train)
        pred_svm = svm.predict(X_test_vec)
        acc_svm = accuracy_score(y_test, pred_svm)

        # 7. KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_vec, y_train)
        pred_knn = knn.predict(X_test_vec)
        acc_knn = accuracy_score(y_test, pred_knn)

        # 8. Tampilkan & simpan hasil
        print("\n=== HASIL AKURASI ===")
        print(f"Naive Bayes: {acc_nb:.2%}")
        print(f"SVM        : {acc_svm:.2%}")
        print(f"KNN        : {acc_knn:.2%}")

        f.write("\n=== HASIL AKURASI ===\n")
        f.write(f"Naive Bayes: {acc_nb:.2%}\n")
        f.write(f"SVM        : {acc_svm:.2%}\n")
        f.write(f"KNN        : {acc_knn:.2%}\n")

        # 9. Laporan klasifikasi Naive Bayes
        f.write("\n=== Laporan Klasifikasi Naive Bayes ===\n")
        f.write(classification_report(y_test, pred_nb))

        # 10. Grafik
        model_names = ['Naive Bayes', 'SVM', 'KNN']
        accuracies = [acc_nb, acc_svm, acc_knn]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(model_names, [x * 100 for x in accuracies])
        plt.ylim(0, 100)
        plt.ylabel("Akurasi (%)")
        plt.title("Perbandingan Akurasi Model")

        # Tambah label di atas bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                     f"{height:.2f}%", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig("grafik_akurasi.png")  # Simpan grafik
        plt.show()

if __name__ == "__main__":
    main()
