# B.Sc.FinalProject
Image Processing Camera Interface for PC Games
Arad Vazani, Ofek Ben Ami
Advisors: Prof. Tammy Riklin-Raviv.

Rehabilitation after injury often involves advanced computer-based exercises, typically limited to clinical settings. Due to the ongoing war, rehabilitation centers are overwhelmed, highlighting the need for effective home-based solutions. This project extends computer-assisted rehabilitation by enabling patients to engage in therapeutic gaming from home. Unlike conventional tools like elastic bands or springs, our system uses modern computer vision techniques. We apply image processing and deep learning to detect human presence, track motion through hand gestures or objects, and map movements to interactive game controls. By utilizing a standard webcam, our solution transforms personal computers into rehabilitation platforms, improving accessibility for patients and reducing pressure on rehabilitation departments.
The game allows the user to choose between two modes of play:
i.	Playing using hand or object recognition
The system initialization is performed by sampling color and building statistical models for the hand and the object using three color channels that provide the most informative data. Afterwards, real-time classification between hand and object is carried out based on the color models, utilizing deep learning to enable seamless integration with the game. 
ii.	Playing using hand gestures recognition
The system recognizes hand gestures in real time and is designed to support physical rehabilitation processes using a simple webcam and open-source tools. A user-friendly interface identifies and classifies gestures by extracting 21 key points with MediaPipe and constructing a 42-element feature vector. The system lets users record gestures for training, applies an SVM classifier for recognition, and delivers real-time visual feedback through a graphical interface during therapeutic or gaming activities. 
The system includes a user interface and mouse integration, enabling gesture recognition to translate user movements into in-game commands by controlling the mouse. 
Keywords: Rehabilitation, Home-based therapy, Computer vision, Image processing, Deep learning, Human motion tracking, Interactive games, Assistive technology, Webcam interface, Physiotherapy.

