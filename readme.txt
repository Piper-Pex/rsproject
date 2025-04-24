
####### For prediction  with CPU only #####################
u need create a conda enviroment with python 3.9  (conda create -n rsproject python==3.9)

u can install by (pip install requirements.txt) directly

OR  Install in the following order

numpy==1.19.5
pandas==1.4.4
tensorflow-cpu==2.6.0
pip uninstall protobuf -y
pip install "protobuf<4.0.0"
pip install scipy==1.7.3
pip install scikit-learn==1.0.2
pip install keras==2.6.0
################################################


I have trained the model and can use the prediction part directly which only uses CPU.
################ for training with GPU, you need ##############################
cuda toolkit 11.8 (i cant run my cuda in conda enviroment, so i just training in my base enviroment)
numpy==1.19.5
pandas==1.4.4
tensorflow-gpu==2.6.0
pip uninstall protobuf -y
pip install "protobuf<4.0.0"
pip install scipy==1.7.3
pip install scikit-learn==1.0.2
pip install keras==2.6.0
################################################################################

#################For Perdiction###########################
Run GMF_predict.py or MLP_predict.py

After running the program for 30 seconds, the user input interface will appear and you can enter multiple music which you like.
(i input some classic musics: Moonlight Sonata , Romeo And Juliet Ballet , Serenade No. 13 , Nocturne For Piano , Swan Lake - Ballet Suite , Gymnopedie #2)

It will search the metadata for music with the corresponding keywords, and list the music it finds. Then you need to enter the corresponding number of the music you want.
(i input 1.1 2.1 3.1 4.1 5.1 6.1)

Finally, the results will be output, and the recommended music is sorted according to the predicted play times.
Then you can go to YouTube to check if the music suits your taste.
