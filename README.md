# NeuroBiometric: Eye Blink Based Biometric Authentication System Using Event-based Neuromorphic Vision Sensor

## Introduction
This repository has released the original dataset of NeuroBiometric.To study the biometric authentication system with an event-based neuromorphic vision sensor, we collected a new dataset with eye blink signals recorded by the DAVIS sensor named NeuroBiometric dataset. 
The DAVIS sensor we used is DAVIS346, which has a resolution of 346*260 pixels, a temporal resolution of 1 μs and an outstanding dynamic range (up to 140 dB). We have 45 volunteers (of whom, 23 are men and 22 are women) to participate in our recording. 

The dataset could be downloaded by https://share.weiyun.com/LoMK32g7.


## Dataset Collection
All the volunteers are in a normal psychological and physiological state. 
Only the facial region includes eyebrow and ocular is recorded by the DAVIS sensor. Additional 60 seconds of break time after every 120 seconds recording is to avoid unusual eye blink due to potential fatigue and distraction.

## Naming
All the raw data are named in a format of 'Davis346redColor-yyyy-mm-ddTsxx-kk.aedat', in which 'yyyy-mm-dd' is the date when a piece of data was collected, 'xx' is the participant id, and 'kk' represents the vedio id of a participant. For an instance, 'Davis346redColor-2019-03-11Ts01-01.aedat' is the 1st vedio of participant 1, which recorded at '2019 May 11th'.

## How to Use
'eyeblink_to_img.py' helps to convert original '.aedat' data into images. 
A filteration process(introduced in our paper) is contained, but you can choose to not filter the raw data by set the 'filter_flag' as False.


## Contact
xiaodingyuan.tj@gmail.com
