import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def index():
    filepath_dict = {'yelp': '../data/data_distinct.csv'}

    df_list = []
    for source, filepath in filepath_dict.items():
        df = pd.read_csv(filepath, names=['label', 'sentence'], sep=',')
        df['source'] = source  # Add another column filled with the source name
        df_list.append(df)

    df = pd.concat(df_list)
    print(df.iloc[0])

    df_yelp = df[df['source'] == 'yelp']

    sentences = df_yelp['sentence'].values
    print(sentences)
    y = df_yelp['label'].values
    print("=========================================")
    print(y)

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)

    print("Accuracy:", score)
    for source in df['source'].unique():
        df_source = df[df['source'] == source]
        sentences = df_source['sentence'].values
        y = df_source['label'].values
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, y, test_size=0.25, random_state=1000)

        vectorizer = CountVectorizer()
        vectorizer.fit(sentences_train)
        X_train = vectorizer.transform(sentences_train)
        X_test = vectorizer.transform(sentences_test)

        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print('Accuracy for {} data: {:.4f}'.format(source, score))
        input_dim = X_train.shape[1]  # Number of features

        model = Sequential()
        model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train,
                            epochs=100,
                            verbose=False,
                            validation_data=(X_test, y_test),
        batch_size = 10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        plt.style.use('ggplot')
        plot_history(history)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentences_train)

        X_train = tokenizer.texts_to_sequences(sentences_train)
        X_test = tokenizer.texts_to_sequences(sentences_test)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

        print(sentences_train[2])
        print(X_train[2])
        maxlen = 100

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        print(X_train[0, :])
        embedding_dim = 50

        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   input_length=maxlen))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train,
                            epochs=20,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        plot_history(history)
        model = Sequential()
        model.add(layers.Embedding(input_dim=vocab_size,
                                   output_dim=embedding_dim,
                                   input_length=maxlen))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train,
                            epochs=50,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        plot_history(history)
        embedding_dim = 100

        model = Sequential()
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train,
                            epochs=10,
                            verbose=False,
                            validation_data=(X_test, y_test),
                            batch_size=10)
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        plot_history(history)
        # Main settings
        epochs = 20
        embedding_dim = 50
        maxlen = 100
        output_file = '../data/output.txt'
        # Run grid search for each source (yelp, amazon, imdb)
        for source, frame in df.groupby('source'):
            print('Running grid search for data set :', source)
            sentences = df['sentence'].values
            y = df['label'].values

            # Train-test split
            sentences_train, sentences_test, y_train, y_test = train_test_split(
                sentences, y, test_size=0.25, random_state=1000)

            # Tokenize words
            tokenizer = Tokenizer(num_words=5000)
            tokenizer.fit_on_texts(sentences_train)
            X_train = tokenizer.texts_to_sequences(sentences_train)
            X_test = tokenizer.texts_to_sequences(sentences_test)

            # Adding 1 because of reserved 0 index
            vocab_size = len(tokenizer.word_index) + 1

            # Pad sequences with zeros
            X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
            X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

            # Parameter grid for grid search
            param_grid = dict(num_filters=[32, 64, 128],
                              kernel_size=[3, 5, 7],
                              vocab_size=[vocab_size],
                              embedding_dim=[embedding_dim],
                              maxlen=[maxlen])
            model = KerasClassifier(build_fn=create_model,
                                    epochs=epochs, batch_size=10,
                                    verbose=False)
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                      cv=4, verbose=1, n_iter=5)
            grid_result = grid.fit(X_train, y_train)

            # Evaluate testing set
            test_accuracy = grid.score(X_test, y_test)

            # Save and evaluate results
            prompt = input(f'finished {source}; write to file and proceed? [y/n]')
            if prompt.lower() not in {'y', 'true', 'yes'}:
                break
            with open(output_file, 'a') as f:
                s = ('Running {} data set\nBest Accuracy : '
                     '{:.4f}\n{}\nTest Accuracy : {:.4f}\n\n')
                output_string = s.format(
                    source,
                    grid_result.best_score_,
                    grid_result.best_params_,
                    test_accuracy)
                print(output_string)
                f.write(output_string)



index()