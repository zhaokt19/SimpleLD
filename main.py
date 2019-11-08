# 视频lane detection
from moviepy.editor import VideoFileClip
from IPython.display import HTML

from pipeline import lane_img_pipeline

white_output = 'solidWhiteRight_output.mp4'  # output文件名
clip1 = VideoFileClip('solidWhiteRight.mp4')  # 读入input video
print(clip1.fps)  # frames per second 25, 默认传给write
white_clip = clip1.fl_image(lane_img_pipeline)  # 对每一帧都执行lane_img_pipeline函数，函数返回的是操作后的image
white_clip.write_videofile(white_output, audio=False)  # 输出经过处理后的每一帧图片，audio=false,不输出音频

# ipython jupyter notebook网页显示video
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))