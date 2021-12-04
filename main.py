import argparse
import cv2
import numpy as np

class AnchorPoints():
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def generate_anchor_points(self, stride=16, row=3, line=3):
        row_step = stride / row
        line_step = stride / line

        shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
        shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        anchor_points = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        return anchor_points

    # shift the meta-anchor to get an acnhor points
    def shift(self, shape, stride, anchor_points):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel()
        )).transpose()

        A = anchor_points.shape[0]
        K = shifts.shape[0]
        all_anchor_points = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
        all_anchor_points = all_anchor_points.reshape((K * A, 2))

        return all_anchor_points
    def __call__(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = self.generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = self.shift(image_shapes[idx], self.strides[idx], anchor_points)
            all_anchor_points = np.append(all_anchor_points, shifted_anchor_points, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return all_anchor_points.astype(np.float32)

class P2PNet():
    def __init__(self, modelPath, confThreshold=0.5):
        self.model = cv2.dnn.readNet(modelPath)
        self.inputNames = 'input'
        self.outputNames = ['pred_logits', 'pred_points']
        self.confThreshold = confThreshold
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        self.anchor_points = AnchorPoints(pyramid_levels=[3,], row=2, line=2)
    def detect(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
        print(img.shape)
        img = (img.astype(np.float32) / 255.0 - self.mean_) / self.std_

        # Preprocess
        inputBlob = cv2.dnn.blobFromImage(img)
        # Forward
        self.model.setInput(inputBlob, self.inputNames)
        outputBlob = self.model.forward(self.outputNames)
        # self.model.setInput(inputBlob)
        # outputBlob = self.model.forward(self.model.getUnconnectedOutLayersNames())
        anchor_points = self.anchor_points(inputBlob)
        output_coord = outputBlob[1] + anchor_points
        points = output_coord[outputBlob[0] > self.confThreshold]
        scores = outputBlob[0][outputBlob[0] > self.confThreshold]

        ratioh, ratiow = srcimg.shape[0]/img.shape[0], srcimg.shape[1]/img.shape[1]
        points[:, 0] *= ratiow
        points[:, 1] *= ratioh
        return scores, points

if __name__=='__main__':
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    parser.add_argument('--imgpath', default='imgs/demo1.jpg', type=str,
                        help="image path")
    parser.add_argument('--onnx_path', default='SHTechA.onnx',
                        help='path where the onnx file saved')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Filter out faces of confidence < conf_threshold.')
    args = parser.parse_args()

    srcimg = cv2.imread(args.imgpath)
    net = P2PNet(args.onnx_path, confThreshold=args.conf_threshold)
    scores, points = net.detect(srcimg)
    print('have', points.shape[0], 'people')
    for i in range(points.shape[0]):
        cv2.circle(srcimg, (int(points[i, 0]), int(points[i, 1])), 2, (0, 0, 255), -1)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()