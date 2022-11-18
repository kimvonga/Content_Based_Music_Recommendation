[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kimvonga/Content_Based_Music_Recommendation/main)

# Content-Based Music Recommendation

Most music recommendation systems rely on user-generated data to form recommendations, e.g. collaborative filtering uses information about one's listening patterns to generate recommendations for another user. While this approach generally works well, it performs poorly for new music or less popular music for which there is less user-generated data. Instead of relying on user-generated data, this project uses waveform-level data to form a content-based music recommendation system that is able to recommend new music and less popular music to users with equal confidence as more popular music. 

This project uses a deep learning architecture to learn genres and genre weights from musical pitch, timbre, and other musical descriptors. The training set comes from the Million Song Dataset. While the project could be refined and improved, the network correctly identifies the genres of some songs. A demonstration of how the network may be used for music recommendations can be found in "Content_Based_Summary.ipynb." A summary of the project, network architecture, as well as samples of code can also be found in the same notebook.
