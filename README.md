# IDL-team-7: Listen, Attend, and Spell: Transcript Generation in Anime

Repository of the final project of Team 7 for 11785 Introduction to Deep Learning S21

In the project, the team attempted to build an end-to-end speaker labeled transcript generation model. The training data for the project is obtained from the Anime Movie Kimi no Na wa.

## Data: 
1. log spectrogram of KNNW original audio soundtrack
2. Labeled original transcript

## Model:
1. Modified LAS model for speech recognition (with transfer learning)
2. CNN-LSTM model for speaker identification 

## Performance:
1. Achieved an average Lev distance of 15.27 for speech recognition
2. Achieved an average classification accuracy of 57% for speaker identification

## How to run:
1. Download the zipped code
2. Use the KNNW_end2end.ipynb for training and generating result
3. Modify the source code loading section as need.
4. Initialize training sessions with different parameters.
