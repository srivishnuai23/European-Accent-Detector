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
