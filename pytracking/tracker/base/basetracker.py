import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import time
import os
import numpy as np
from PIL import Image

class BaseTracker:
    """Base class for all trackers."""

    def __init__(self, params):
        self.params = params

    def initialize(self, image, state, class_info=None):
        """Overload this function in your tracker. This should initialize the model."""
        raise NotImplementedError

    def track(self, image):
        """Overload this function in your tracker. This should track in the frame and update the model."""
        raise NotImplementedError

    def track_sequence(self, sequence):
        """Run tracker on a sequence."""

        # Initialize
        image = self._read_image(sequence.frames[0])

        times = []
        start_time = time.time()
        self.sequence_name = sequence.name
        self.initialize(image, sequence.init_state, init_mask=sequence.init_mask)
        init_time = getattr(self, 'time', time.time() - start_time)
        times.append(init_time)

        if self.params.visualization:
            self.init_visualization()
            self.visualize(image, sequence.init_state)

        # Track
        tracked_bb = [sequence.init_state]
        tracked_conf = [1]
        idx = 1
        for frame in sequence.frames[1:]:
            image = self._read_image(frame)

            start_time = time.time()
            state, conf = self.track(image)
            times.append(time.time() - start_time)

            tracked_bb.append(state)
            tracked_conf.append(conf)

            if self.params.visualization:
                self.visualize(image, state, idx=idx)
            idx += 1

        return tracked_bb, times, tracked_conf

    def track_videofile(self, videofilepath, optional_box=None):
        """Run track with a video file input."""

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)
        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, list, tuple)
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            self.initialize(frame, optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                return

            frame_disp = frame.copy()

            # Draw box
            state = self.track(frame)
            state = [int(s) for s in state]
            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                self.initialize(frame, init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def track_webcam(self):
        """Run tracker with webcam."""

        class UIControl:
            def __init__(self):
                self.mode = 'init'  # init, select, track
                self.target_tl = (-1, -1)
                self.target_br = (-1, -1)
                self.mode_switch = False

            def mouse_callback(self, event, x, y, flags, param):
                if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
                    self.target_tl = (x, y)
                    self.target_br = (x, y)
                    self.mode = 'select'
                    self.mode_switch = True
                elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
                    self.target_br = (x, y)
                elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
                    self.target_br = (x, y)
                    self.mode = 'track'
                    self.mode_switch = True

            def get_tl(self):
                return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

            def get_br(self):
                return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

            def get_bb(self):
                tl = self.get_tl()
                br = self.get_br()

                bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
                return bb

        ui_control = UIControl()
        cap = cv.VideoCapture(0)
        display_name = 'Display: ' + self.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        if hasattr(self, 'initialize_features'):
            self.initialize_features()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            frame_disp = frame.copy()

            if ui_control.mode == 'track' and ui_control.mode_switch:
                ui_control.mode_switch = False
                init_state = ui_control.get_bb()
                self.initialize(frame, init_state)

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)
            elif ui_control.mode == 'track':
                state = self.track(frame)
                state = [int(s) for s in state]
                cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                             (0, 255, 0), 5)

            # Put text
            font_color = (0, 0, 0)
            if ui_control.mode == 'init' or ui_control.mode == 'select':
                cv.putText(frame_disp, 'Select target', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            elif ui_control.mode == 'track':
                cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
                cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ui_control.mode = 'init'

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

    def reset_tracker(self):
        pass

    def press(self, event):
        if event.key == 'p':
            self.pause_mode = not self.pause_mode
            print("Switching pause mode!")
        elif event.key == 'r':
            self.reset_tracker()
            print("Resetting target pos to gt!")

    def init_visualization(self):
        # plt.ion()
        '''
        debug 1: vis RGB and bbox
              2: vis RGB and Correlation map
              3: vis RGB, Correlation Map, GaussianNewtonCG
              4: vis RGB, Depth,
              5: vis RGB, depth, mask
        '''
        self.pause_mode = False

        if self.params.debug == 5:
            self.fig, ((self.ax, self.ax_d), \
                       (self.ax_initmask, self.ax_initmaskimg), \
                       (self.ax_rgb_patches, self.ax_d_patches), \
                       (self.ax_m, self.ax_mrgb)) = plt.subplots(4, 2)

        elif self.params.debug == 4:
            self.ax_m = None
            self.fig, (self.ax, self.ax_d) = plt.subplots(1, 2)
        else:
            self.ax_d = None
            self.ax_m = None
            self.fig, self.ax = plt.subplots(1)

        # self.fig.canvas.manager.window.move(800, 50)
        self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (100, 50))

        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.tight_layout()


    def visualize(self, image, state, idx=0):
        if isinstance(image, dict) and self.params.debug >= 4:
            color, depth = image['color'], image['depth']
            im_h, im_w, im_c = color.shape
            self.ax.cla()
            self.ax.imshow(color)
            self.ax_d.cla()
            depth[depth > self.max_depth] = self.max_depth
            depth[depth < self.min_depth] = self.min_depth
            depth = (depth - self.min_depth) / (self.max_depth - self.min_depth)
            depth = np.asarray(depth*255, dtype=np.uint8)
            depth = cv.applyColorMap(depth, cv.COLORMAP_JET)
            self.ax_d.imshow(depth)
        else:
            if isinstance(image, dict):
                image = image['color']
                im_h, im_w, im_c = image.shape
            self.ax.cla()
            self.ax.imshow(image)
        self.ax.set_title('Frame %d'%idx)

        if self.params.debug == 5:
            self.ax_m.cla()
            self.ax_m.imshow(self.mask)
            self.ax_m.set_title('predicted mask')

            # masked_img = np.uint8(self.masked_img)
            masked_img = self.masked_img
            masked_img[np.all(masked_img == (0, 0, 0), axis=-1)] = (255,255,255)
            # masked_img = Image.fromarray(np.uint8(masked_img)).convert('RGBA')

            if self.attn_dcf is not None:
                # attn_dcf = self.attn_dcf.clone().detach().cpu().numpy().squeeze()
                #
                attn_dcf = self.attn_dcf.squeeze()

                self.ax_mrgb.cla()
                self.ax_mrgb.imshow(attn_dcf)
                self.ax_mrgb.set_title('dcf Depth')

            # self.ax_mrgb.cla()
            # self.ax_mrgb.imshow(masked_img)
            # self.ax_mrgb.set_title('predicted mask over rgb')


            self.ax_initmask.cla()
            self.ax_initmask.imshow(self.init_mask)
            self.ax_initmask.set_title('init mask')


            init_masked_img = np.uint8(self.init_masked_img)
            init_masked_img[np.all(init_masked_img == (0, 0, 0), axis=-1)] = (255,255,255)
            self.ax_initmaskimg.cla()
            self.ax_initmaskimg.imshow(init_masked_img)
            self.ax_initmaskimg.set_title('init mask over rgb')

            if self.rgb_patches is not None:
                self.ax_rgb_patches.cla()
                rgb_score = Image.fromarray(np.uint8(self.rgb_patches)).convert('RGBA')
                if self.score_map is not None:
                    cm = plt.get_cmap('jet')
                    colored_image = cm(self.score_map)
                    scoremap = Image.fromarray((colored_image[:, :, :3]*255).astype(np.uint8)).convert('RGBA')
                    scoremap = scoremap.resize(rgb_score.size)
                    rgb_score = Image.blend(rgb_score, scoremap, 0.3)

                self.ax_rgb_patches.imshow(rgb_score)
                self.ax_rgb_patches.set_title('RGB patch, Conf :%.02f'%self.conf_)

                self.ax_d_patches.cla()
                self.ax_d_patches.imshow(self.d_patches)
                try:
                    self.ax_d_patches.set_title('D patch: %d'%self.prev_target_depth)
                except:
                    pass

            # Song
            if self.polygon is not None:
                polygon = patches.Polygon(self.polygon, closed=True, facecolor='none', edgecolor='r')
                self.ax_m.add_patch(polygon)

        if len(state) == 4:
            state[0] = max(state[0], 0)
            state[1] = max(state[1], 0)
            if state[0]+state[2] >= im_w:
                state[2] = im_w - state[0] - 1
            if state[1]+state[3] >= im_h:
                state[3] = im_h - state[1] - 1
            pred = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=2, edgecolor='r', facecolor='none')
            pred_d = patches.Rectangle((state[0], state[1]), state[2], state[3], linewidth=2, edgecolor='r', facecolor='none')
        elif len(state) == 8:
            p_ = np.array(state).reshape((4, 2))
            pred = patches.Polygon(p_, linewidth=2, edgecolor='r', facecolor='none')
            pred_d = patches.Polygon(p_, linewidth=2, edgecolor='r', facecolor='none')
        else:
            print('Error: Unknown prediction region format.')
            exit(-1)

        self.ax.add_patch(pred)

        if self.params.debug == 5:
            self.ax_d.add_patch(pred_d)

        if hasattr(self, 'gt_state') and False:
            gt_state = self.gt_state
            rect = patches.Rectangle((gt_state[0], gt_state[1]), gt_state[2], gt_state[3], linewidth=1, edgecolor='g',
                                     facecolor='none')
            self.ax.add_patch(rect)

        self.ax.set_axis_off()
        self.ax.axis('equal')

        self.ax_d.set_axis_off()
        self.ax_d.axis('equal')

        plt.draw()
        plt.pause(0.00001)

        if self.pause_mode:
            plt.waitforbuttonpress()

    def _read_image(self, image_file: str):
        if isinstance(image_file, dict):
            # For CDTB and DepthTrack RGBD datasets
            color = cv.cvtColor(cv.imread(image_file['color']), cv.COLOR_BGR2RGB)
            depth = cv.imread(image_file['depth'], -1)
            depth = np.nan_to_num(depth)
            images = {'color':color, 'depth':depth}
            return images
        else:
            return cv.cvtColor(cv.imread(image_file), cv.COLOR_BGR2RGB)
