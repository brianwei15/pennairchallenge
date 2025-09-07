How to Run the Code:
Task 1: run detect_shapes.py

Task 2: run the command: python detect_shapes_video_task2.py --preview 
or just run detect_shapes_video_task2.py if you dont want a preview while it is running.

Task 3: run the command: python detect_shapes_video.py --preview
or just run detect_shapes_video.py if you dont want a preview while it is running.


Task 1 Approach:
I used detect_shapes.py to solve this task.

Since I read all the challenges before attempting this one, I knew that I wanted to develop an algorithm that could generalize between all shapes and backgrounds. I saw that in later tasks, objects start to become hard to differentiate from the background, so I decided to apply a Gaussian blur on the background, and for each pixel, detemine how close it is to the average color of the background. If it is above a threshold, then it is a part of a shape. 

Image: refer to Task1Solution.png

Task 2 Explanation:

Approach: Since my previous algorithm in task 1 had trouble detecting the green trapezoid, I knew I needed to add another filter. I decided to use the motion-based background subtraction (MOG2) because it would be able to detect objects that are moving that couldn't be detected based off of color contrast alone. This would be super helpful for the next task as well.

Report: The algorithm traced the objects accurately throughout the video. Initially, when the algorithm was untuned, there were moments where patches of grass would get detected by the color filter when the yellow circle leaves the frame. I decided to isolate the color filter alone in detect_shapes_video_mask_test.py and tuned the Gaussian blur values and color difference threshold to where that did not happen anymore. Afterwards, everything worked smoothly. The one issue with using MOG2 is that it requires a certain number of frames of history to determine what is moving and what is not. So, in the beginning of the video, some objects leave a small trail behind because of it.

The most imporant improvement I made to my code in this step was to make the blur constants easily adjustable, so my code would be more easily tunable.

Video: https://www.youtube.com/watch?v=Jm-pelGKAH8


Task 3 Explanation:
I kept the algorithm relatively the same from task 2, barring a couple of constants. The big challenge for me was dealing with the pentagon, which moved at slow speeds and had a difficult dark color shade. Since the MOG2 motion-based background subtraction takes in a certain number of past frames to determine what has moved, making this number lower helped the filter be more sensitive to slow change like the pentagon at the beginning and the end of the video. 

After implementing the tunable variables for blur and color difference threshold last task, tuning the color filter wasn't that bad. I also utilized detect_shapes_video_mask_test.py to isolate the color mask for easier tuning.


Video: https://www.youtube.com/watch?v=l2yLELk5nHE

