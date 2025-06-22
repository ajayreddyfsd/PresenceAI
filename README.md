# PresenceAI: Real-Time Public Speaking Analyzer

Effective communication is more than just words — it’s about how you present yourself physically. Non-verbal cues like posture, gestures, and eye contact can greatly influence how your message is perceived. PresenceAI leverages advanced computer vision and AI techniques to track and evaluate your posture, hand gestures, and eye contact during live or recorded speeches. It acts as a real-time performance coach to help you improve your delivery. By quantifying key non-verbal behaviors and combining them with AI-generated feedback, PresenceAI helps you build confidence, increase engagement, and deliver more impactful presentations.

# Techs Used: 

MediaPipe is used to detect precise body and hand landmarks, enabling accurate analysis of posture and meaningful gestures. This helps differentiate expressive movement from random hand waving.

OpenCV handles the video stream—either live from a webcam or from pre-recorded footage—ensuring frame-by-frame processing for smooth and consistent analysis.

MongoDB stores all collected metrics, making it easy to track progress over time, compare sessions, and build analytics dashboards if needed.


