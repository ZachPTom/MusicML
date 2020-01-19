import pickle
import numpy
import glob
import csv
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation


def produce():
    """ Produce a song """
    # load notes from training
    with open('output/notes_durations', 'rb') as fp:
        notes = pickle.load(fp)

    # Get elements
    elements = sorted(set(item for item in notes))
    n_elements = len(set(notes))

    model_input, normalized_input = produce_sequences(notes, elements, n_elements)

    # sort weight files for mass production
    weight_files = []
    for wf in glob.glob("*.hdf5"):
        weight_files.append(wf)

    weight_files = sorted(weight_files)

    # iterate through all weights to produce multiple songs
    for file in weight_files:
        model = generate_model(normalized_input, n_elements, file)
        model_output = produce_notes(model, model_input, elements, n_elements)
        prediction_analysis(model_output, file)
        produce_midi(model_output, file)


def produce_sequences(notes, elements, n_elements):
    """ Produce sequences to be fed into the model """
    # map between elements and integers
    element_to_int = dict((note, number) for number, note in enumerate(elements))

    sequence_length = 100
    model_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        model_input.append([element_to_int[char] for char in sequence_in])
        output.append(element_to_int[sequence_out])

    n_elements = len(model_input)

    # reshape input for LSTM
    normalized_input = numpy.reshape(model_input, (n_elements, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_elements)

    return model_input, normalized_input


def generate_model(model_input, n_elements, file):
    """ build the model """
    print("Generating RNN from %s" % file)

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

    # Load the weights from training
    model.load_weights(file)

    return model


def produce_notes(model, model_input, elements, n_elements):
    """ produce notes from the model """
    # random sequence input
    start = numpy.random.randint(0, len(model_input) - 1)

    int_to_element = dict((number, note) for number, note in enumerate(elements))

    element = model_input[start]
    model_output = []

    # produce 500 notes
    for i in range(500):
        model_input = numpy.reshape(element, (1, len(element), 1))
        model_input = model_input / float(n_elements)

        prediction = model.predict(model_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_element[index]
        model_output.append(result)

        element.append(index)
        element = element[1:len(element)]

    return model_output


def prediction_analysis(model_output, file):
    """Determine variation statistics of the produced music"""

    prediction_file = open("model_output_durations.txt", "a")

    # write prediction output to a file to visualize notes
    prediction_file.write("\n\n" + file + "\n\n")
    output_list = [f' {x} ' for x in model_output]
    prediction_file.writelines(output_list)
    prediction_file.close()

    note_dict = dict((note, 0) for note in model_output)

    # count number of times each unique element appears in prediction
    # count total number of chords
    # count total number of rests
    # count sequential variation
    chord_count = 0
    rest_count = 0
    seq_variation_count = 0
    for index, element in enumerate(model_output):
        note_dict[element] += 1
        if (':' in element) or element.isdigit():
            chord_count += 1
        if 'r' in element:
            rest_count += 1
        # if note is different than the two before and two after
        if 1 < index < len(model_output) - 2:
            pitch = element.split('~')
            if pitch[0] != model_output[index - 1].split('~')[0] and pitch[0] != model_output[index - 2].split('~')[0] and \
                    pitch[0] != model_output[index + 1].split('~')[0] and pitch[0] != model_output[index + 2].split('~')[0]:
                seq_variation_count += 1

    # count total number of elements that are not the mode element
    maximum = max(note_dict, key=note_dict.get)
    count = 0
    for note in note_dict:
        if note != maximum:
            count += note_dict[note]

    # calculations
    tot_variation_pct = count / len(model_output)
    tot_chord_pct = chord_count / len(model_output)
    tot_rest_pct = rest_count / len(model_output)
    seq_variation_pct = seq_variation_count / len(model_output)

    with open('variation_percentages_durations.csv', mode='a') as vp_file:
        vp_writer = csv.writer(vp_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        vp_writer.writerow([file, tot_variation_pct, seq_variation_pct, tot_chord_pct, tot_rest_pct])


def produce_midi(model_output, file):
    """ convert to notes and make a midi file """
    offset = 0
    output_notes = []

    # create note, chord, rests with note durations
    for element in model_output:
        # chord
        if (':' in element) or element[0].isdigit():
            notes_in_chord = element.split(':')
            notes = []
            dur = ''
            for current_note in notes_in_chord:
                if '~' in current_note:
                    # last element of chord will have duration appended
                    get_duration = current_note.split('~')
                    new_note = note.Note(int(get_duration[0]))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                    dur = get_duration[1]
                else:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = float(dur)
            output_notes.append(new_chord)
        # rest
        elif 'r' in element:
            new_rest = note.Rest()
            new_rest.offset = offset
            output_notes.append(new_rest)
        # note
        else:
            get_duration = element.split('~')
            new_note = note.Note(get_duration[0])
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = float(get_duration[1])
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=f'test_output_{file[0:17]}.mid')


if __name__ == '__main__':
    produce()
