# opticalflow
Lucas-Kanade implementation of Optical Flow using Python, for personal education purposes

Example results in k13 (Window size 13), k7 (Window size 7), and lktracker videos

Notable issues:
- Tracks over a field of view despite being more for local tracking, todo is to use Harris corner detection to choose features to track 
