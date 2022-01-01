# opticalflow
Lucas-Kanade implementation of Optical Flow using Python, for personal education purposes

Notable issues:
- Tracks over a field of view despite being more for local tracking, todo is to use Harris corner detection to choose features to track 
- Doesn't account for aperture problem yet (large eigenvalues of the structure tensor A)
