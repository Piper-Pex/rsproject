
### For prediction  with CPU only
You first need to download the models file to the project root directory
https://drive.google.com/file/d/1FHq_vi2ccZm07lN1TXy4FhDt3-nkjvFD/view?usp=sharing
https://drive.google.com/file/d/1SOXnmNGDQUxg_A7URuOJQG3cwoXfHCjr/view?usp=sharing


u need create a conda enviroment with python 3.9  (conda create -n rsproject python==3.9)

u can install by (pip install requirements.txt) directly

OR  Install in the following order
  

```
pip install numpy==1.19.5
pip install pandas==1.4.4
pip install tensorflow-cpu==2.6.0
pip uninstall protobuf -y
pip install "protobuf<4.0.0"
pip install scipy==1.7.3
pip install scikit-learn==1.0.2
pip install keras==2.6.0
```
################################################

### For training with GPU, you need
User rating data files for training (prediction part does not need to be downloaded), you need to download it to the project directory
(Download Link: http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip)

cuda toolkit 11.8
```
pip install numpy==1.19.5
pip install pandas==1.4.4
pip install tensorflow-gpu==2.6.0
pip uninstall protobuf -y
pip install "protobuf<4.0.0"
pip install scipy==1.7.3
pip install scikit-learn==1.0.2
pip install keras==2.6.0
```
################################################################################

### Prediction progress
1. Run GMF_predict.py or MLP_predict.py or Fusion_predict.py

2. After running the program for 30 seconds, the user input interface will appear and you can enter multiple music which you like.
(i input some musics: The Darkest Night Of All   , Put Your Records On , Please_ Before I Go, Kräne)

3. It will search the metadata for music with the corresponding keywords, and list the music it finds. Then you need to enter the corresponding number of the music you want.
(i input 1.1 2.1 3.1 4.1)

4. Finally, the results will be output, and the recommended music is sorted according to the predicted play times.
Then you can go to YouTube to check if the music suits your taste.



### Some playlist in the Dataset
You can find more user-music-plays record in the `ShowSomePlaylist.ipynb`, 
and you can change the "ShowPlaylists_num" to show more play list

User 093cb74eb3c517c5179ae24caf0ebec51b24d2a2 Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • Corrido de Boxeo — Ry Cooder — play 9667 times
  • Crash — Matt Willis — play 494 times
  • Lola Leave Your Light On — Gov't Mule — play 243 times
  • Que Triste Es Decir Adios — Yndio — play 158 times
  • Find You Waiting — DecembeRadio — play 152 times
  • Bubbly — Colbie Caillat — play 128 times
  • XXX's And OOO's (An American Girl) — Trisha Yearwood — play 121 times
  • Maybe — Sick Puppies — play 110 times
  • DONTTRUSTME [BENNYBLANCOREMIX] FEATURINGKIDCUDI (Explicit Bonus Version) — 3OH!3 — play 87 times
  • Andina — Strunz & Farah — play 69 times

User 119b7c88d58d0c6eb051365c103da5caf817bea6 Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • Se Dagen Kom — Kari Hansa & Gregers Hes — play 227 times
  • Pojo Pojo — Cyberfit — play 217 times
  • Absolution: Of Flight and Failure — A Hope For Home — play 168 times
  • Protect Your Mind 2009 (Braveheart) — Darren Bailie — play 167 times
  • Weekends And Bleak Days [Hot Summer] (album version) — The Young Knives — play 158 times
  • Night Song — Nina Simone — play 148 times
  • Jamaica Roots II(Agora E Sempre) — Natiruts — play 142 times
  • Am I A Fool — Sense Field — play 142 times
  • Searchin' — Brant Bjork — play 132 times
  • When You Wish Upon A Star — Louis Armstrong — play 131 times

User 3fa44653315697f42410a30cb766a4eb102080bb Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • California Dreamin' — The Mamas & The Papas — play 216 times
  • Unwell (Album Version) — matchbox twenty — play 203 times
  • Stay — Lisa Loeb & Nine Stories — play 196 times
  • Message In A Bottle — The Police — play 193 times
  • Message In A Bottle — The Police — play 193 times
  • Crazy Little Thing Called Love (Album Version) — Michael Bublé — play 146 times
  • Groove Me — King Floyd — play 133 times
  • Breakfast At Tiffany's — Deep Blue Something — play 125 times
  • Under Pressure — Queen — play 125 times
  • Girls Just Want To Have Fun — Cyndi Lauper — play 113 times

User a2679496cd0af9779a92a13ff7c6af5c81ea8c7b Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • What You Know — Two Door Cinema Club — play 159 times
  • Magic Carpet Ride — Steppenwolf — play 121 times
  • Fliegende Fische — Pohlmann. — play 114 times
  • Horn Concerto No. 4 in E flat K495: II. Romance (Andante cantabile) — Barry Tuckwell/Academy of St Martin-in-the-Fields/Sir Neville Marriner — play 88 times
  • Something Good Can Work — Two Door Cinema Club — play 75 times
  • L.E.S. Artistes — Santogold — play 74 times
  • Revelry — Kings Of Leon — play 73 times
  • The Big Gundown — The Prodigy — play 68 times
  • Dice — Finley Quaye;Beth Orton — play 68 times
  • Undo — Björk — play 68 times

User 4ae01afa8f2430ea0704d502bc7b57fb52164882 Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • Mike_ Aaron And Eddie — Haiku D'Etat — play 164 times
  • Holly Hobby — Casiotone For The Painfully Alone — play 126 times
  • None Shall Pass (Main) — Aesop Rock — play 107 times
  • Transparency — White Denim — play 94 times
  • My Glorious — Delirious? — play 94 times
  • Nothing Better (Album) — Postal Service — play 93 times
  • Hold On — Holy Ghost — play 89 times
  • Friendship Train — Gladys Knight & The Pips — play 89 times
  • New Day — Bouncing Souls — play 84 times
  • The Gift — Angels and Airwaves — play 79 times

User 281deab3afccc906251ef67a8eda2b9f9baec459 Listened（Play times>=3 times ）Musics（Play times desending ordering）：
  • Unite (2009 Digital Remaster) — Beastie Boys — play 171 times
  • My Love — Justin Timberlake — play 153 times
  • My Love — Justin Timberlake — play 153 times
  • Revelry — Kings Of Leon — play 142 times
  • Canada — Five Iron Frenzy — play 141 times
  • Monster — Lady GaGa — play 110 times
  • Samba De Una Nota So´ — Joa~o Gilberto — play 107 times
  • Sincerité Et Jalousie — Alliance Ethnik — play 102 times
  • Better That We Break — Maroon 5 — play 78 times
  • Starry Eyed Surprise (Album Version) — Shifty — play 78 times