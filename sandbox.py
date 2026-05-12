# Run this in terminal first:
# pip install librosa soundfile --break-system-packages 

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# --- PART 1: FEATURE EXTRACTION ---
def extract_accent_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=5.0)
        y = librosa.util.normalize(y)

        mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta   = librosa.feature.delta(mfccs)
        delta2  = librosa.feature.delta(mfccs, order=2)
        chroma  = librosa.feature.chroma_stft(y=y, sr=sr)
        contrast= librosa.feature.spectral_contrast(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        # CHANGE: Added zero crossing rate and RMS energy.
        # WHY: ZCR captures how noisy vs tonal a voice is (useful for consonant-heavy accents).
        #      RMS captures energy patterns — some accents are naturally more dynamic.
        zcr     = librosa.feature.zero_crossing_rate(y)
        rms     = librosa.feature.rms(y=y)

        def stats(f):
            return np.hstack([np.mean(f, axis=1), np.std(f, axis=1)])

        return np.hstack([
            stats(mfccs), stats(delta), stats(delta2),
            stats(chroma), stats(contrast),
            np.mean(rolloff), np.std(rolloff),
            np.mean(zcr),    np.std(zcr),
            np.mean(rms),    np.std(rms)
        ])
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None


# --- PART 2: DATA LOADING ---
def load_dataset(data_path):
    file_paths, labels = [], []
    classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    print(f"Indexing Accents: {classes}")

    for accent_label in classes:
        folder_path = os.path.join(data_path, accent_label)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.mp3', '.wav')):
                file_paths.append(os.path.join(folder_path, filename))
                labels.append(accent_label)

    counts = Counter(labels)
    print(f"\nSamples per class: {dict(counts)}")
    valid_classes = {cls for cls, count in counts.items() if count >= 5}
    filtered = [(f, l) for f, l in zip(file_paths, labels) if l in valid_classes]
    file_paths, labels = zip(*filtered) if filtered else ([], [])

    train_files, test_files, y_train, y_test = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    def extract(files, file_labels):
        X, y = [], []
        for path, label in zip(files, file_labels):
            vec = extract_accent_features(path)
            if vec is not None:
                X.append(vec)
                y.append(label)
        return np.array(X), np.array(y)

    X_train, y_train = extract(train_files, y_train)
    X_test, y_test   = extract(test_files,  y_test)
    return X_train, X_test, y_train, y_test


# --- PART 3: TRAINING ---
def train_accent_model(data_folder):
    X_train, X_test, y_train, y_test = load_dataset(data_folder)
    if len(X_train) == 0:
        print("No training data found.")
        return None

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)}...")
    print(f"Feature vector size: {X_train.shape[1]}")

    # CHANGE: Switched from a single GradientBoostingClassifier to a VotingClassifier
    # ensemble of three different algorithms.
    # WHY: With only ~77 training samples, any single model is unreliable. Each model
    #      makes different kinds of mistakes — voting them together averages out those
    #      errors and produces a more robust prediction.
    #
    #   GradientBoosting — strong on complex non-linear patterns
    #   SVM (RBF kernel) — excels at high-dimensional feature spaces like ours (~200 features)
    #   KNN              — simple but often surprisingly effective with scaled audio features
    #
    # 'soft' voting uses predicted probabilities rather than just majority vote,
    # which makes better use of the confidence each model has in its answer.

    gbc = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ))
    ])

    svm = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
    ])

    knn = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
    ])

    model = VotingClassifier(
        estimators=[('gbc', gbc), ('svm', svm), ('knn', knn)],
        voting='soft'
    )

    # CHANGE: Switched to StratifiedKFold with only 5 folds (was also 5, but now
    # explicitly shuffled) to be safer with small class sizes.
    print("\nRunning 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred))
    return model


# --- PART 4: INTERACTIVE FILE SELECTOR ---
def select_test_file(data_dir):
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in sorted(files):
            if f.endswith(('.mp3', '.wav')):
                full_path = os.path.join(root, f)
                label = os.path.relpath(full_path, data_dir)
                audio_files.append((full_path, label))

    audio_files.sort(key=lambda x: x[1])

    print("\n" + "="*55)
    print("         SELECT AN AUDIO FILE TO TEST")
    print("="*55)

    if audio_files:
        print("\nFiles found in your dataset:\n")
        for i, (path, label) in enumerate(audio_files, 1):
            print(f"  [{i:>2}] {label}")

    print(f"\n  [C]  Enter a custom file path")
    print(f"  [Q]  Quit")
    print("\n" + "-"*55)

    while True:
        choice = input("Your choice: ").strip()

        if choice.upper() == 'Q':
            return None

        if choice.upper() == 'C':
            custom = input("Enter full path to audio file: ").strip()
            if os.path.exists(custom):
                if custom.endswith(('.mp3', '.wav')):
                    return custom
                else:
                    print("  ✗ File must be a .mp3 or .wav file. Try again.")
            else:
                print("  ✗ File not found. Try again.")
            continue

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(audio_files):
                return audio_files[idx][0]
            else:
                print(f"  ✗ Please enter a number between 1 and {len(audio_files)}.")
            continue

        print("  ✗ Invalid input. Enter a number, C, or Q.")


# --- PART 5: PREDICT ON SELECTED FILE ---
def predict_file(model, file_path, data_dir):
    rel   = os.path.relpath(file_path, data_dir)
    parts = rel.split(os.sep)

    # CHANGE: Only show true label comparison when the file is inside a named subfolder.
    # WHY: Test files in a flat folder (e.g. /tests/test_spanish.mp3) have no subfolder
    #      to infer an accent from, so the old code showed "True label: .. ✗ WRONG"
    #      which was meaningless and confusing.
    in_subfolder = len(parts) > 1 and not parts[0].startswith('..')
    true_label   = parts[0] if in_subfolder else None

    print(f"\nAnalysing: {os.path.basename(file_path)}")
    feat = extract_accent_features(file_path)

    if feat is None:
        print("Could not extract features from this file.")
        return

    prediction = model.predict([feat])[0]
    probs       = model.predict_proba([feat])[0]
    classes     = model.classes_
    ranked      = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

    print("\n" + "="*40)
    print("         PREDICTION RESULTS")
    print("="*40)
    print(f"  Predicted accent : {prediction}")

    # Only show true label line if we actually know it
    if true_label:
        match = "✓ CORRECT" if prediction == true_label else "✗ WRONG"
        print(f"  True label       : {true_label}  {match}")
    else:
        print(f"  True label       : Unknown (file outside labelled dataset)")

    print(f"\n  Confidence breakdown:")
    for accent, prob in ranked:
        bar = "█" * int(prob * 30)
        print(f"    {accent:<15} {prob*100:5.1f}%  {bar}")
    print("="*40)


# --- MAIN ---
if __name__ == "__main__":
    DATA_DIR  = "/home/codio/workspace/data/"
    DATA_TEST = "/home/codio/workspace/tests/"

    if not os.path.exists(DATA_DIR):
        print("Error: 'data/' folder missing!")
    else:
        accent_model = train_accent_model(DATA_DIR)

        if accent_model:
            while True:
                chosen_file = select_test_file(DATA_TEST)

                if chosen_file is None:
                    print("\nGoodbye!")
                    break

                predict_file(accent_model, chosen_file, DATA_TEST)

                again = input("\nTest another file? (Y/N): ").strip().upper()
                if again != 'Y':
                    print("\nGoodbye!")
                    break