export FRAMERATE=150
python resample_reconstructions.py -i ../outputs/EvAid/playball/reconstruction -o ../outputs/EvAid/playball/resampled -r $FRAMERATE
ffmpeg -framerate $FRAMERATE -i ../outputs/EvAid/playball/resampled/frame_%010d.png ../outputs/EvAid/playball/video_"$FRAMERATE"Hz.mp4
python resample_reconstructions.py -i ../outputs/EvAid/playball/e2vid_reconstruction -o ../outputs/EvAid/playball/e2vid_resampled -r $FRAMERATE
ffmpeg -framerate $FRAMERATE -i ../outputs/EvAid/playball/e2vid_resampled/frame_%010d.png ../outputs/EvAid/playball/e2vid_video_"$FRAMERATE"Hz.mp4
