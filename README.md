# European-Accent-Detector README
Initially my KSU CC590 Final Project using a python AI model to learn English accents from across Europe to something more put together. 
This project was done together with Reid Ellis, who contributed to collecting and cleaning audio samples from the GMU data set. (Accents Source: https://accent.gmu.edu/browse_language.php?function=find&language=english)

The following code refers to iteration 1 and 2, which uses the GMU data set.

--- 
In case code breaks because its missing librosa package, run this command in the console:
pip install librosa soundfile --break-system-packages


## Run instructions:
- type "python3 accent.py" in the terminal and hit enter
- Wait 5-10 minutes (yes, its slow), until the interface with options shows up
- type the number of the corresponding audio file you want the model to guess the country is from
- If you want to add your own audio, add it to the /tests folder

---

# Iteration 1 and 2 Report

European Accents Detecting Model
Ried Ellis and Srivishnu Chandramouli
Problem 
In Europe multiple languages exist in close proximity to each other, which can cause confusion. Often people who grew up speaking their country’s native language learn other languages, primarily English, so that they can communicate in other countries. However when both people have strong accents it may feel like they are still speaking completely different languages. Software that could quickly detect the native language of any given speaker, could allow for more rapid access to translation services. In this case an artificial intelligence could be beneficial due to its skill in pattern recognition. 
Method
As this is an expansion of Project 11, Accent Recognition, we used the same data set, maintained by the GMU Speech Accent Archive (1), which hosts roughly 30 second clips of a standardized speech uploaded by users. The audios are organized by native language, birth place, age, and many more factors. 
Project 11 trained an accent detection model from 2 locations (US, and UK) with 10 audio samples each. The model’s accuracy ranged from 60-70%. 
In our expansion, we decided to focus on Europe specifically and included the following 10 languages and their respective sample count (due to unavailability, not all accents had 30):
Comparisons to Project 11
UML Comparisons 



Edited Version 




Sample Count Iteration 1
Project 11 Accent List
Project 11 Sample Count
Modified Accent List
Modified Sample Count
United Kingdom (UK)
10
Norwegian 
7
English
10
French
10




German
10




Italian
10




Polish
10




Portuguese
10




Romanian
10




Serbian
10




Spanish
10




Swedish
10


Sample Count Iteration 2
Project 11 Accent List
Project 11 Sample Count
Modified Accent List
Modified Sample Count
United Kingdom (UK)
10
United Kingdom (UK)
30
English
10
French
30




German
30




Italian
30




Polish
30




Portuguese
30




Romanian
24




Serbian
19




Spanish
30




Swedish
23


Data Comparison Between Iteration 1 and 2


10 Samples/Class (est.)
30 Samples/Class (est.)
Total Samples
100
300
Training Set (80%)
80
240
Training per class
8
24
CV fold training size
6-7
19-20

Model Feature Comparison
Project 11
Modified
13 features of speech to distinguish between accents
15 features: 13 original + 2 more:
Zero crossing rate
RMS energy

Better placement of consonant heavy accents and accents with highly dynamic speaking patterns. 
No testing feature within console application
Added a console feature to test accuracy quickly. 1 sample per accent available to check the accuracy of the model.


Considerations when selecting Data
For the sake of this program only the first two features were used, where 30 audios from each language were taken from speakers with primarily European birthplaces (e.g. French speakers from Europe and French Speakers from Africa would likely be too different). 
Some languages did not have a full 30 speakers in the archive, but it functioned as the best balance of having enough data and avoiding skew.
	The base for this program was pulled from the 11th lab in the Kansas State University 




Results and Interpretation
We measured model accuracy based on testing each of the sample accents against the model. After 10 rounds, the average percentage was the iteration’s accuracy percentage. 
Iteration 1
We reached the maximum improvement possible by improving the model within our understanding. Accuracy so low, the model was as good as random guessing. 

Testing Log

Round
Correct (%)
1
10
2
20
3
10
4
10
5
20
6
10
7
10
8
10
9
10
10
10
Average
12.0%


Iteration 2
No changes to the model were made, but we increased the samples from 10 -> 30. As mentioned before, not all accents have 30 samples. However the increased data greatly improved the models accuracy:

Testing Log

Round
Correct (%)
1
40
2
60
3
60
4
60
5
40
6
40
7
50
8
40
9
40
10
40
Average
47.0%


Reflection
When increasing the scope of the project from 2 accents to 10, the model became very inaccurate, being only slightly better than guessing. This was with only 10 speakers per language however, as seen in iteration one. Increasing the number of audios had a clear and significant improvement on the accuracy of the model. As such expanding the number of audios that are used in training data would be critical in improving accuracy further. Alongside that, we believe that the inconsistent number of speakers per accent likely causes the model to unfairly not choose those languages; reducing the skew by having a greater number of Serbian speakers for example would cause significant improvements in recognizing Serbian speakers. However, just adding more speakers will have diminishing returns, as compute times expand and become too expensive to justify with more speakers. We were already seeing this with the difference between iteration 1 and 2 nearly doubling the time to train. As such we would also like to use a better distributed data set. Ideally, the dataset would have an equal distribution of speakers across age, gender, and geographically within the European country. The Speech Archive used in this lab had access to many of these features, but we were unable to take them into account when choosing audios, as we were struggling to even get enough audios from Europe. A larger number of speakers from Europe would allow us to be much more selective with the other features that modify their accent, and build a more robust and representative training set, even if not every one of those speakers would be used in training. We believe that these changes would allow the model to more accurately characterize what speech traits are distinct to the country, rather than placing all female voices in France, as a majority of French speakers in training were French. 
Resources
https://accent.gmu.edu/browse_language.php

AI Use
Generated UMLs using ChatGPT, code is original

