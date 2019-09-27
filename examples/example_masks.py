from nyan.collections import Video, Sequential


EXAMPLE_VIDEO = '../static/nyan.mp4'
EXAMPLE_MASKS = '../static/segmentations/*.png'

if __name__ == '__main__':
    # load the video in debug mode
    video = Video(fp=EXAMPLE_VIDEO, debug_mode=True)

                                          # perform some arbitrary pre-processing:
    video = video[:, 128:-128]            # . crop 128 pixels off of the left and right of the video
    video.rotate(-45.)                    # . rotate video by -45 degrees
    video.zoom((1.1, 1.))                 # . zoom into video by 10% in the y-direction
    video.convert_color(('H', 'S', 'V'))  # . convert the video to HSV colour space

    # show the image history (this is useful for debugging pre-processing steps)
    video.debug()

    # load some segmentation masks for the video (let's not do debug mode this time)
    masks = Sequential(directory_wildcard=EXAMPLE_MASKS)

    # show the first mask and the first image so we can see what they look like
    video.show(0)
    masks.show(0)

    # transform the masks into the space of the pre-processed videos
    masks_mapped = video.transform_image(masks, 'start', 'end')
    masks_mapped.show()

    # transform the masks_mapped back into the original space
    masks_mapped_back_again = video.transform_image(masks_mapped, 'end', 'start')
    masks_mapped_back_again.show()
