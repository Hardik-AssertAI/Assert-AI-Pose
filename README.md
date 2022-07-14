# Assert-AI-Pose

Pose estimation, as the name suggests is the estimation of some poses, pertaining to it's use cases from an image or video (sources can vary).<br>
Generally in the Computer Vision environment, most tasks are done using pre-existing CNN architecture, but instead of this we strived to achieve similar results using a much different approach.<br>

We mainly used an open source library https://google.github.io/mediapipe{Mediapipe}, which identifies some key landmarks on the general human body, and this information about the landmarks can be used and manipulated to get some interesting results.<br>

Using these landmarks obtained, tracking a human in the image or video is pretty easy, and is done fairly accurately.<br>
It also works in real time very well, as it does real-time tracking quite well on standard webcam videos as well, along with the standard tracking on recorded videos.<br>

Individual elements of the landmarks can be accessed too. This estimation also gives a fair sense of depth as it tracks it in a 3D virtual environment, giving the appropriate [x,y,z] co-ordinates.<br>

Note that by default these are normalized from the centre of the hip to take care of the difference in scales, image sizes, resolutions, etc. <br>
The real time processing runs decently fast on CPU's unlike some other libraries which are much slower, need GPU's and are complex to implement.<br>

NOTE: The only visible downside was when parts of the human body overlapped significantly, the corresponding landmarks would be less accurate.<br>
