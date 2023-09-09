# !pip install https://github.com/pyannote/pyannote-audio/archive/refs/heads/develop.zip
# !pip install ipython==7.34.0
# !pip install SpeechRecognition
# !pip install pocketsphinx
# !pip install pydub


# Importing libraries
from pyannote.audio import Pipeline
from pyannote.database.util import load_rttm
from pydub import AudioSegment
import pandas as pd
import speech_recognition as sr
import streamlit as st
import os

from keras.models import model_from_json
import pickle
import librosa
import numpy as np
import streamlit as st


def diarize(input):
    audio_file_path = sample_path + input
    audio_name = input.split('.')[0]

    # Create a folder with the audio name
    output_folder = audio_name + '-output'
    os.makedirs(output_folder, exist_ok=True)

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
    # Perform diarization and dump the diarization output to disk using RTTM format
    diarization = pipeline(audio_file_path,num_speakers=2)
    output_path = audio_name + ".rttm"
    with open(output_path, "w") as rttm:
        diarization.write_rttm(rttm)    

    # Get dataframe
    def rttm_to_dataframe(rttm_file_path):
        columns = ["Type", "File ID","Channel","Start Time","Duration","Orthography","Confidence","Speaker","x","y"]
        with open(rttm_file_path,"r") as rttm_file:
            lines = rttm_file.readlines()
        data = []
        for line in lines:
            line = line.strip().split()
            data.append(line)
        df = pd.DataFrame(data, columns = columns)
        df = df.drop(["x","y","Orthography","Confidence"],axis=1)
        return df
    
    
    def extract_text_from_audio(audio_file_path, start_time, end_time):
        r = sr.Recognizer()
        with sr.AudioFile(audio_file_path) as source:
            audio = r.record(source, duration = end_time, offset = start_time)
        text = r.recognize_google(audio)
        return text
    
    rttm_file_path = output_path
    df = rttm_to_dataframe(rttm_file_path)
    df = df.astype({'Start Time':'float'})
    df = df.astype({'Duration':'float'})
    df['Utterance'] = None
    df['End Time'] =  df['Start Time'] + df['Duration']

    for ind in df.index:
        start_time = df['Start Time'][ind]
        end_time = df['End Time'][ind]
        try:
            transcription = extract_text_from_audio(audio_file_path, start_time, end_time)
            df['Utterance'][ind] = transcription
        except:
            df['Utterance'][ind] = 'Not Found'
    df = df.drop(["Type","File ID","Channel"],axis=1)


    def save_audio_segment(audio_file_path, start_time, end_time, output_file_path):
        sound = AudioSegment.from_wav(audio_file_path)
        segment = sound[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
        segment.export(output_file_path, format='wav')

    for index, row in df.iterrows():
        start_time = row['Start Time']
        end_time = row['End Time']
        output_file_path = output_folder +'/'+ f"segment_{index:02d}.wav"  # You can customize the output file name
        save_audio_segment(audio_file_path, start_time, end_time, output_file_path)

    df.insert(1,'End Time',df.pop('End Time'))
    return df

output_folder = "output-audios/"

def show_outputs(df, audio_name):
    output_folder = audio_name + "-output"
    output_dir_list = os.listdir(output_folder)

    file_path = []
    for file in output_dir_list:
        file_path.append(output_folder +'/'+ file)

    output_path_df = pd.DataFrame(file_path, columns=['Path'])
    emotions = []
    for loc in output_path_df['Path']:
        # Check if loc is a directory and skip it
        if os.path.isdir(loc):
            print(f"Skipping directory: {loc}")
            continue
        pred_emotion = prediction(loc)
        emotions.append(pred_emotion)

    pred_emotions_df = pd.DataFrame(emotions, columns=['Emotion'])
    newdf = pd.concat([df, pred_emotions_df], axis=1)
    return newdf


def model():
    json_file = open("C:/Users/hanaa/GIT/emotion-recognition/Speech/outputs/CNN_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("C:/Users/hanaa/GIT/emotion-recognition/Speech/outputs/best_model1_weights.h5")
    return loaded_model

def scaler():
    with open("C:/Users/hanaa/GIT/emotion-recognition/Speech/outputs/scaler2.pickle", 'rb') as f:
        scaler2 = pickle.load(f)
    return scaler2

def encoder():
    with open("C:/Users/hanaa/GIT/emotion-recognition/Speech/outputs/encoder2.pickle", 'rb') as f:
        encoder2 = pickle.load(f)
    return encoder2

def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr)) #stacking horizontally
    
    # MFCC 
    mfcc = np.mean(librosa.feature.mfcc(y=data).T, axis=0)
    result = np.hstack((result, mfcc))
    
    # Root Mean Square value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))
    
    return result


def get_predict_feat(path):
    d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
    res=extract_features(d)
    result=np.array(res)
    result=np.reshape(result,newshape=(1,22))
    i_result = scaler().transform(result)
    final_result=np.expand_dims(i_result, axis=2)
    return final_result


def prediction(path1):
    res=get_predict_feat(path1)
    predictions=model().predict(res)
    y_pred = encoder().inverse_transform(predictions)
    emotion = y_pred[0][0]  
    return emotion


def button_click():
    st.header('Speaker Diarization')
    inp = st.selectbox('Select an audio sample', ['Brooklyn 99', 'Gilmore Girls 01', 'Gilmore Girls 02', 'Home Alone', 'La La Land', 'Love and Other Drugs', 'New Girl 01', 'New Girl 02', 'Notebook', 'The Office 01', 'The Office 02'])
    input_audio = audio_mapping[inp]
    audio_name = input_audio.split('.')[0]
    if st.button('Go'):
        st.write(inp)
        st.audio(sample_path+input_audio)
        output_df = diarize(input_audio)     
        final_df = show_outputs(output_df, audio_name)
        st.dataframe(final_df)



# default_path="C:/Users/hanaa/Documents/hanan/DUK/SEM2/mini project/data/"
sample_path="datasets/wav/"
audio_mapping = {'Brooklyn 99':"brooklyn-sample.wav", 
                 'Gilmore Girls 01':"gilmore-01-sample.wav", 
                 'Gilmore Girls 02':"gilmore-02-sample.wav", 
                 'Home Alone':"homealone-sample.wav",
                 'La La Land':"lalaland-sample.wav", 
                 'Love and Other Drugs':"laod-sample.wav", 
                 'New Girl 01':"newgirl-01-sample.wav", 
                 'New Girl 02':"newgirl-02-sample.wav", 
                 'Notebook':"notebook-sample.wav", 
                 'The Office 01':"office-01-sample.wav", 
                 'The Office 02':"office-02-sample.wav"}



button_click()