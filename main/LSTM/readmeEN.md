[Русский](https://github.com/kilsense/Text-Classification/blob/7b821631ea7b598579efacdb8587ab23753875b4/main/LSTM/readme.md)

This folder contains my first classification program that I made with a very serious face.

# char_classifier_trainer.py
This file contains the complete training and validation system for the LSTM classifier.  
To run it, you only need to specify the data file.

### Configuration
At the beginning of the code, there is a block with parameters that you can play around with. Here is a brief overview of the main ones:
- `MAX_VOCAB_SIZE` — the maximum size of the dictionary. If there is a lot of text, you can limit it to frequent characters so as not to inflate the model.
- `MAX_SEQ_LENGTH` — the maximum length of a string (in characters). Anything longer is simply truncated.
- `EMBEDDING_DIM` — the dimension of the vector for each character. The larger it is, the more accurate the differences, but the higher the memory load.
- `HIDDEN_DIM` — the size of the LSTM hidden state, i.e., how much information the model “remembers” about the context. The larger it is, the deeper the understanding, but the slower the training.
- `NUM_LAYERS` — the number of LSTM layers. One layer is simpler and faster, two or more are more accurate but heavier.
- `BATCH_SIZE` — how many examples are processed in one step. More is faster, less is more stable.
- `LEARNING_RATE` — learning rate. If too high, the model will “jump”; if too low, it will hardly learn at all.
- `NUM_EPOCHS` — how many times the model goes through all the data. Usually 3–5 is enough to understand if it works at all.
- `TEST_SIZE` — the proportion of data set aside for accuracy testing (validation). 0.2 = 20% of the data goes into the test.
- `MIN_SAMPLES` — the minimum number of examples per class. Classes with fewer examples are simply ignored so as not to interfere with training.

### A little about the data
For training, I used an export from Tatoeba.  
The file had a `.csv` extension, but inside it was separated by tabs (as in `.tsv`), so in the code the separation is done by tabs.  
The first column is the class label, the second is the text.  

Because there are classes with very few examples in Tatoeba, I added the `MIN_SAMPLES` parameter. It allows you to exclude such classes so that the model does not “stumble” on them.

### A little about saving
The dictionary is saved **together with the model**, so the `VOCAB_SAVE_PATH` parameter is not technically used in the code.  
An example of how to get the dictionary from the saved model is shown in the `using.py` file.

# using.py
This file is responsible for using trained models.  
It can load the model and classify texts.  

Texts are also truncated to a specified length — this can be removed, but it is better not to do so if the model was trained with truncation.

There are two modes:
1. **Interactive** — you can enter text manually, and the program will show up to 10 most likely classes.
2. **Batch** — you can load a CSV file:
   - either just a list of lines (one line = one text),
   - or the same format as during training (`class<TAB>text`) so that the program can compare actual labels with predictions.

# dataset_editor.html
This file is a GUI for convenient editing of datasets. It uses the same format as the rest of the project. It displays some statistics about the set, and you can add and search for records.
There may be problems with data sets that are too large.
The data in this GUI is stored in the browser's memory, so you can reload and close the page without worrying about the data (but it's better to play it safe and export it before closing).
Import
