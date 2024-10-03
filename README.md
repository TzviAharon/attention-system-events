This project is part of my B.Sc. seminar, where I implemented an algorithm described in [this paper](https://ieeexplore.ieee.org/abstract/document/10100741)
. The algorithm automatically detects the center of attention in a scene based on event data.

For the event data, I used the [Zurich Event Camera Dataset](https://rpg.ifi.uzh.ch/davis_data.html).

To run the project, ensure that the following items are in the same directory as main.py:

  A folder named images containing the event data.
  
  The following files:
    
    events.txt
    
    images.txt
    
    calib.txt


The algorithm processes event-based data to track and highlight the most prominent zones in a scene, focusing on the center of attention dynamically.
