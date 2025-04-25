
I have trained the model and you can use the prediction part directly which only uses CPU.

####### For prediction  with CPU only #####################
You first need to download the model file to the project root directory
https://drive.google.com/file/d/1FHq_vi2ccZm07lN1TXy4FhDt3-nkjvFD/view?usp=sharing
https://drive.google.com/file/d/1SOXnmNGDQUxg_A7URuOJQG3cwoXfHCjr/view?usp=sharing


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


################ for training with GPU, you need ##############################
User rating data files for training (prediction part does not need to be downloaded), you need to download it to the project directory
(Download Link: http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip)

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
(i input some musics: The Darkest Night Of All   , Put Your Records On , Please_ Before I Go, Kräne)

It will search the metadata for music with the corresponding keywords, and list the music it finds. Then you need to enter the corresponding number of the music you want.
(i input 1.1 2.1 3.1 4.1)

Finally, the results will be output, and the recommended music is sorted according to the predicted play times.
Then you can go to YouTube to check if the music suits your taste.

#################Some playlist in the Dataset ############################
you can find more user-music-plays record in the "ShowSomePlaylist.ipynb" , 
and you can change the "ShowPlaylists_num" to show more play list

用户 093cb74eb3c517c5179ae24caf0ebec51b24d2a2 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Corrido de Boxeo — Ry Cooder — 播放 9667 次
  • Crash — Matt Willis — 播放 494 次
  • Lola Leave Your Light On — Gov't Mule — 播放 243 次
  • Que Triste Es Decir Adios — Yndio — 播放 158 次
  • Find You Waiting — DecembeRadio — 播放 152 次
  • Bubbly — Colbie Caillat — 播放 128 次
  • XXX's And OOO's (An American Girl) — Trisha Yearwood — 播放 121 次
  • Maybe — Sick Puppies — 播放 110 次
  • DONTTRUSTME [BENNYBLANCOREMIX] FEATURINGKIDCUDI (Explicit Bonus Version) — 3OH!3 — 播放 87 次
  • Andina — Strunz & Farah — 播放 69 次

用户 119b7c88d58d0c6eb051365c103da5caf817bea6 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Se Dagen Kom — Kari Hansa & Gregers Hes — 播放 227 次
  • Pojo Pojo — Cyberfit — 播放 217 次
  • Absolution: Of Flight and Failure — A Hope For Home — 播放 168 次
  • Protect Your Mind 2009 (Braveheart) — Darren Bailie — 播放 167 次
  • Weekends And Bleak Days [Hot Summer] (album version) — The Young Knives — 播放 158 次
  • Night Song — Nina Simone — 播放 148 次
  • Jamaica Roots II(Agora E Sempre) — Natiruts — 播放 142 次
  • Am I A Fool — Sense Field — 播放 142 次
  • Searchin' — Brant Bjork — 播放 132 次
  • When You Wish Upon A Star — Louis Armstrong — 播放 131 次

用户 3fa44653315697f42410a30cb766a4eb102080bb 听过的（播放>=3次）歌曲（按播放总数排序）：
  • California Dreamin' — The Mamas & The Papas — 播放 216 次
  • Unwell (Album Version) — matchbox twenty — 播放 203 次
  • Stay — Lisa Loeb & Nine Stories — 播放 196 次
  • Message In A Bottle — The Police — 播放 193 次
  • Message In A Bottle — The Police — 播放 193 次
  • Crazy Little Thing Called Love (Album Version) — Michael Bublé — 播放 146 次
  • Groove Me — King Floyd — 播放 133 次
  • Breakfast At Tiffany's — Deep Blue Something — 播放 125 次
  • Under Pressure — Queen — 播放 125 次
  • Girls Just Want To Have Fun — Cyndi Lauper — 播放 113 次

用户 a2679496cd0af9779a92a13ff7c6af5c81ea8c7b 听过的（播放>=3次）歌曲（按播放总数排序）：
  • What You Know — Two Door Cinema Club — 播放 159 次
  • Magic Carpet Ride — Steppenwolf — 播放 121 次
  • Fliegende Fische — Pohlmann. — 播放 114 次
  • Horn Concerto No. 4 in E flat K495: II. Romance (Andante cantabile) — Barry Tuckwell/Academy of St Martin-in-the-Fields/Sir Neville Marriner — 播放 88 次
  • Something Good Can Work — Two Door Cinema Club — 播放 75 次
  • L.E.S. Artistes — Santogold — 播放 74 次
  • Revelry — Kings Of Leon — 播放 73 次
  • The Big Gundown — The Prodigy — 播放 68 次
  • Dice — Finley Quaye;Beth Orton — 播放 68 次
  • Undo — Björk — 播放 68 次

用户 4ae01afa8f2430ea0704d502bc7b57fb52164882 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Mike_ Aaron And Eddie — Haiku D'Etat — 播放 164 次
  • Holly Hobby — Casiotone For The Painfully Alone — 播放 126 次
  • None Shall Pass (Main) — Aesop Rock — 播放 107 次
  • Transparency — White Denim — 播放 94 次
  • My Glorious — Delirious? — 播放 94 次
  • Nothing Better (Album) — Postal Service — 播放 93 次
  • Hold On — Holy Ghost — 播放 89 次
  • Friendship Train — Gladys Knight & The Pips — 播放 89 次
  • New Day — Bouncing Souls — 播放 84 次
  • The Gift — Angels and Airwaves — 播放 79 次

用户 281deab3afccc906251ef67a8eda2b9f9baec459 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Unite (2009 Digital Remaster) — Beastie Boys — 播放 171 次
  • My Love — Justin Timberlake — 播放 153 次
  • My Love — Justin Timberlake — 播放 153 次
  • Revelry — Kings Of Leon — 播放 142 次
  • Canada — Five Iron Frenzy — 播放 141 次
  • Monster — Lady GaGa — 播放 110 次
  • Samba De Una Nota So´ — Joa~o Gilberto — 播放 107 次
  • Sincerité Et Jalousie — Alliance Ethnik — 播放 102 次
  • Better That We Break — Maroon 5 — 播放 78 次
  • Starry Eyed Surprise (Album Version) — Shifty — 播放 78 次

用户 d7d2d888ae04d16e994d6964214a1de81392ee04 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Johnny Too Bad — The Slickers — 播放 63 次
  • Sweet and Dandy — The Maytals — 播放 63 次
  • Music My Rock — Bedouin Soundclash — 播放 63 次
  • Born To Win — Jimmy Cliff — 播放 55 次
  • Rivers Of Babylon — The Melodians — 播放 54 次
  • Music Maker — Jimmy Cliff — 播放 53 次
  • On My Life — Jimmy Cliff — 播放 53 次
  • Pressure Drop — The Maytals — 播放 52 次
  • My World Is Blue — Jimmy Cliff — 播放 50 次
  • Slave To Love (1999 Digital Remaster) — Bryan Ferry — 播放 48 次

用户 3325fe1d8da7b13dd42004ede8011ce3d7cd205d 听过的（播放>=3次）歌曲（按播放总数排序）：
  • So Glad To See You — Hot Chip — 播放 155 次
  • Roll On — Dntel — 播放 152 次
  • High (Album Version) — James Blunt — 播放 139 次
  • Better Together — Jack Johnson — 播放 134 次
  • Put Your Records On — Corinne Bailey Rae — 播放 114 次
  • Girl — Beck — 播放 96 次
  • Please_ Before I Go — Derek Webb — 播放 83 次
  • Modern Nature — Sondre Lerche — 播放 81 次
  • Young Folks — Peter_ Bjorn and John Featuring Victoria Bergsman — 播放 79 次
  • If It's Love — Train — 播放 77 次

用户 6b36f65d2eb5579a8b9ed5b4731a7e13b8760722 听过的（播放>=3次）歌曲（按播放总数排序）：
  • The Darkest Night Of All — Lisa Germano — 播放 2165 次
  • Now You're Gone — Basshunter Feat. DJ Mental Theos Bazzheadz — 播放 1646 次
  • Whip Game Proper [feat. Lil' Wayne] (Explicit Album Version) — Twista featuring Lil' Wayne — 播放 427 次
  • Falling Inside The Black (Album Version) — Skillet — 播放 218 次
  • Hero (Album Version) — Skillet — 播放 146 次
  • Walk On Water — Basshunter — 播放 98 次
  • Bittersweet (Crashings Album Version) — Falling Up — 播放 88 次
  • Now You're Gone [DJ Alex Extended Mix] — Basshunter feat. DJ Mental Theos Bazzheadz — 播放 81 次
  • Don't Walk Away — Basshunter — 播放 50 次
  • Day & Night — Basshunter — 播放 30 次

用户 b7c24f770be6b802805ac0e2106624a517643c17 听过的（播放>=3次）歌曲（按播放总数排序）：
  • Hands Up To Heaven (Spray Mix) — Heaven 17 — 播放 45 次
  • Money Vibrations — Little Beaver — 播放 45 次
  • Give It Up (LP Version) — Pepper — 播放 42 次
  • Stormtrooper (LP Version) — Pepper — 播放 42 次
  • Come Into My World — Kylie Minogue — 播放 38 次
  • Sincerité Et Jalousie — Alliance Ethnik — 播放 37 次
  • Tumbleweed — Afroman — 播放 36 次
  • Le Courage Des Oiseaux — Dominique A — 播放 36 次
  • Invalid — Tub Ring — 播放 35 次
  • Fuck Kitty — Frumpies — 播放 35 次
