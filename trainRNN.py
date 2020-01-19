import glob
import pickle
import numpy
import csv
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger


def train_RNN():
    """ Train the RNN to produce music"""
    notes = process_data()

    # get amount of unique elements
    n_elements = len(set(notes))

    model_input, model_output = produce_sequences(notes, n_elements)

    model = generate_model(model_input, n_elements)

    train(model, model_input, model_output)


def process_data():
    """ Get all the notes and chords from the designated midi files in the directory """
    notes = []
    composerFiles = []

    # Obtain filenames of specific composer music
    with open('chopin-maestro-training-v2.0.0.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                composerFiles.append(row[4][5:])
                line_count += 1

        print(f'Processed {line_count} lines.')

    # parse specified midi files
    count = 0
    for file in glob.glob("midi_Files/*.midi"):
        if file[11:] in composerFiles:
            count += 1
            print("Parsing %s" % file)
            midi = converter.parse(file)
            print(f'Processed {count} files.')

            # parse the instrument parts in midi files
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()

            # add duration to notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch) + '~' + str(element.duration.quarterLengthNoTuplets))
                elif isinstance(element, chord.Chord):
                    # append chord notes as list of notes integer values separated by colons
                    notes.append('{0}~{1}'.format(':'.join(str(n) for n in element.normalOrder),
                                                  str(element.duration.quarterLengthNoTuplets)))
                elif isinstance(element, note.Rest):
                    notes.append('r')

    with open('output/notes_durations', 'wb') as filepath:
        # output notes to a file for use by the prediction model
        pickle.dump(notes, filepath)

    return notes


def produce_sequences(notes, n_elements):
    """ Produce sequences to be fed into the model  """
    sequence_length = 100

    # get all elements
    elements = sorted(set(item for item in notes))

    # dictionary mapping elements to numbers
    element_to_int = dict((note, number) for number, note in enumerate(elements))

    model_input = []
    model_output = []

    # produce input sequences and expected outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        model_input.append([element_to_int[char] for char in sequence_in])
        model_output.append(element_to_int[sequence_out])

    n_patterns = len(model_input)

    # reshape input for LSTM
    model_input = numpy.reshape(model_input, (n_patterns, sequence_length, 1))
    model_input = model_input / float(n_elements)

    model_output = np_utils.to_categorical(model_output)

    return model_input, model_output


def generate_model(model_input, n_elements):
    """ Build the RNN """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(model_input.shape[1], model_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_elements))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, model_input, model_output):
    """ train the RNN """
    filepath = "weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=False,
        mode='min'
    )
    csv_logger = CSVLogger('epoch_loss_log_durations.csv', append=True, separator=';')
    callbacks_list = [checkpoint, csv_logger]

    model.fit(model_input, model_output, epochs=100, batch_size=200, callbacks=callbacks_list)


if __name__ == '__main__':
    train_RNN()
