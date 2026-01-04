import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';
import { MiniprogramIOHandler } from '@/plugins/MiniprogramIOHandler';


// const classNameStrs = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush"
const classNameStrs = "人,自行车,汽车,摩托车,飞机,公交车,火车,卡车,船,交通信号灯,消防栓,停车标志,停车计时器,长凳,鸟,猫,狗,马,羊,牛,大象,熊,斑马,长颈鹿,双肩背包,雨伞,手提包,领带,手提箱,飞盘,滑雪板,滑雪板,运动球,风筝,棒球棒,棒球手套,滑板,冲浪板,网球拍,瓶子,红酒杯,杯子,叉子,刀,勺子,碗,香蕉,苹果,三明治,橙子,西兰花,胡萝卜,热狗,比萨饼,甜甜圈,蛋糕,椅子,沙发,盆栽植物,床,餐桌,马桶,电视,笔记本电脑,鼠标,遥控器,键盘,手机,微波炉,烤箱,烤面包机,水槽,冰箱,书,时钟,花瓶,剪刀,泰迪熊,吹风机,牙刷"
const classNames = classNameStrs.split(',')

// 模型期望的固定输入尺寸 [height, width]
const MODEL_INPUT_SIZE = [640, 640]; // [height, width]

export const version = '0.0.1';

/** @docinline */
export type ObjectDetectionBaseModel =
    'yolo11n'|'yolo11m'|'yolo11s';

export interface DetectedObject {
  bbox: [number, number, number, number];  // [x, y, width, height]
  class: string;
  score: number;
}

export interface ModelConfig {
  base?: ObjectDetectionBaseModel;
  modelUrl?: string;
}

export async function load(config: ModelConfig = {}) {
  if (tf == null) {
    throw new Error(
        `Cannot find TensorFlow.js. If you are using a <script> tag, please ` +
        `also include @tensorflow/tfjs on the page before using this model.`);
  }
  const base = config.base || 'yolo11n';
  const modelUrl = config.modelUrl;
  if (['yolo11n', 'yolo11m', 'yolo11s'].indexOf(base) ===
      -1) {
    throw new Error(
        `ObjectDetection constructed with invalid base model ` + `${base}. `);
  }

  const objectDetection = new ObjectDetection(base, modelUrl);
  await objectDetection.load();
  return objectDetection;
}

export class ObjectDetection {
  private modelPath: string | undefined;
  private model: tfconv.GraphModel;

  constructor(base: ObjectDetectionBaseModel, modelUrl?: string) {
    this.modelPath = modelUrl;
  }

  // 预处理函数：将图像保持长宽比缩放到目标尺寸，并用灰色填充空白区域
  private preprocessImage(image: tf.Tensor3D, targetSize: number[]): { image: tf.Tensor4D, metadata: any } {
    const [targetHeight, targetWidth] = targetSize;
    const [imgHeight, imgWidth] = image.shape.slice(0, 2);

    // 计算缩放比例，保持宽高比
    const scale = Math.min(targetHeight / imgHeight, targetWidth / imgWidth);

    // 计算新尺寸
    let newHeight = Math.round(imgHeight * scale);
    let newWidth = Math.round(imgWidth * scale);

    // 计算填充
    let padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;

    // 保持长宽比，最小填充模式
    const dw = targetWidth - newWidth;
    const dh = targetHeight - newHeight;
    padLeft = Math.floor(dw / 2);
    padRight = Math.ceil(dw / 2);
    padTop = Math.floor(dh / 2);
    padBottom = Math.ceil(dh / 2);

    // 调整尺寸并添加填充
    const resized = tf.image.resizeBilinear(image, [newHeight, newWidth]);
    const padded = tf.pad(resized, [
        [padTop, padBottom],
        [padLeft, padRight],
        [0, 0]
    ]);

    // 归一化到 [0, 1]
    const normalized = tf.div(padded, 255.0);

    // 添加批次维度
    const batched = tf.expandDims(normalized);

    // 返回预处理后的图像和元数据
    return {
        image: batched,
        metadata: {
            scale: scale,
            padding: [padLeft, padTop],
            originalSize: [imgHeight, imgWidth],
            targetSize: [targetHeight, targetWidth]
        }
    };
  }

  // 坐标还原函数：将模型输出的坐标转换回原始图像空间
  private scaleBoxes(boxes: number[], metadata: any): number[] {
    const { scale, padding, originalSize } = metadata;
    const [origHeight, origWidth] = originalSize;
    const [padLeft, padTop] = padding;

    // 将坐标从模型输出空间转换回原图空间
    let [x1, y1, x2, y2, confidence, classId] = boxes;

    // 移除填充
    x1 = x1 - padLeft;
    x2 = x2 - padLeft;
    y1 = y1 - padTop;
    y2 = y2 - padTop;

    // 缩放回原图尺寸
    x1 = x1 / scale;
    x2 = x2 / scale;
    y1 = y1 / scale;
    y2 = y2 / scale;

    // 限制在原图范围内
    x1 = Math.max(0, Math.min(x1, origWidth));
    x2 = Math.max(0, Math.min(x2, origWidth));
    y1 = Math.max(0, Math.min(y1, origHeight));
    y2 = Math.max(0, Math.min(y2, origHeight));

    return [x1, y1, x2, y2, confidence, classId];
  }

  async load() {
    if (this.modelPath) {
      this.model = await tfconv.loadGraphModel(this.modelPath);
    } else {
      console.error("模型URL未指定")
//       const ioHandler = new MiniprogramIOHandler(this.modelPath);
//       this.model = await tfconv.loadGraphModel(ioHandler);
    }

    const [targetHeight, targetWidth] = MODEL_INPUT_SIZE;
    const zeroTensor = tf.zeros([1, targetHeight, targetWidth, 3], 'float32');
    // console.log('detectTestData=', `[1, ${targetHeight}, ${targetWidth}, 3]`);
    // Warmup the model.
    const result = await this.model.executeAsync(zeroTensor) as tf.Tensor[];
    // console.log('detectTestResult=', result);

    await Promise.all(result.map(t => t.data()));
    result.map(t => t.dispose());
    zeroTensor.dispose();
  }

  /**
   * Infers through the model.
   *
   * @param img The image to classify. Can be a tensor or a DOM element image,
   * video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   * @param minScore The minimum score of the returned bounding boxes
   * of detected objects. Value between 0 and 1. Defaults to 0.5.
   */
  private async infer(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxNumBoxes: number, minScore: number): Promise<DetectedObject[]> {

    // 获取原始图像尺寸
    let originalWidth: number;
    let originalHeight: number;

    const [targetHeight, targetWidth] = MODEL_INPUT_SIZE;

    let imgTensor: tf.Tensor3D;
    if (!(img instanceof tf.Tensor)) {
      // DOM元素或ImageData直接获取尺寸
      if ('width' in img && 'height' in img) {
        originalWidth = img.width;
        originalHeight = img.height;
      } else if (img instanceof ImageData) {
        originalWidth = img.width;
        originalHeight = img.height;
      } else {
        // 其他类型，默认尺寸
        originalWidth = targetWidth;
        originalHeight = targetHeight;
      }

      imgTensor = tf.browser.fromPixels(img) as tf.Tensor3D;  // 转换为[0,255]范围的张量
    } else {
      // 从张量的shape中获取尺寸
      imgTensor = img as tf.Tensor3D;
      const shape = imgTensor.shape;
      originalHeight = shape[0];
      originalWidth = shape[1];
    }

    console.log('原始图像尺寸(宽*高):', originalWidth, 'x', originalHeight);
    console.log('推理图像尺寸(宽*高):', targetWidth, 'x', targetHeight);

    let preprocessedResult;
    const batched = tf.tidy(() => {
      // 使用类的预处理方法
      preprocessedResult = this.preprocessImage(imgTensor, MODEL_INPUT_SIZE);
      return preprocessedResult.image;
    });

    // model returns two tensors:
    // 1. detections with shape of [1, 300, 6]
    // 2. 辅助信息 with shape of [1500]
    // where 300 is the number of box detectors, 6 is [x1, y1, x2, y2, conf, class]
    const result = await this.model.executeAsync(batched) as tf.Tensor[];

    console.log("yolon11模型推理结果 result=", result);
    let detections;
    let detectNum;
    if (result[0] && result[0].shape[1]) {
        detections = result[0].dataSync() as Float32Array;
        detectNum = result[0].shape[1];
    } else if (result[1] && result[1].shape[1]) {
        detections = result[1].dataSync() as Float32Array;
        detectNum = result[1].shape[1];
    } else {
        //todo
    }



    // clean the webgl tensors
    batched.dispose();
    tf.dispose(result);

    // const prevBackend = tf.getBackend();
    // run post process in cpu
    // if (tf.getBackend() === 'webgl') {
    //   tf.setBackend('cpu');
    // }

    const detectedObjects: DetectedObject[] = [];
    const uniqueDetections = new Set<string>();  // 用于去重的Set

    // 遍历检测结果并应用过滤条件
    for (let i = 0; i < detectNum && detectedObjects.length < maxNumBoxes; i++) {
        const idx = i * 6;
        const box = [
          detections[idx],
          detections[idx + 1],
          detections[idx + 2],
          detections[idx + 3],
          detections[idx + 4],
          detections[idx + 5]
        ];

        // 使用类的坐标还原方法
        const [x1, y1, x2, y2, conf, classId] = this.scaleBoxes(box, preprocessedResult.metadata);

        // 应用置信度过滤
        if (conf > minScore) {
            // 对坐标和置信度进行两位小数处理
            const roundedConf = parseFloat(conf.toFixed(2));
            const roundedX1 = parseFloat(x1.toFixed(4));
            const roundedY1 = parseFloat(y1.toFixed(4));
            const roundedX2 = parseFloat(x2.toFixed(4));
            const roundedY2 = parseFloat(y2.toFixed(4));

            // 生成唯一key：classId + conf + x1 + y1 + x2 + y2
            const uniqueKey = `${classId}_${roundedConf}_${roundedX1}_${roundedY1}_${roundedX2}_${roundedY2}`;

            // 去重处理：只有当key不存在时才添加到结果中
            if (!uniqueDetections.has(uniqueKey)) {
                uniqueDetections.add(uniqueKey);

                const bbox: [number, number, number, number] = [
                    roundedX1,
                    roundedY1,
                    roundedX2 - roundedX1,
                    roundedY2 - roundedY1
                ];

                detectedObjects.push({
                    class: classNames[classId] || '未知',
                    score: roundedConf,
                    bbox: bbox
                });
            }
        }
    }

    // restore previous backend
    // if (prevBackend !== tf.getBackend()) {
    //   tf.setBackend(prevBackend);
    // }

    return detectedObjects;
  }

  /**
   * Detect objects for an image returning a list of bounding boxes with
   * associated class and score.
   *
   * @param img The image to detect objects from. Can be a tensor or a DOM
   *     element image, video, or canvas.
   * @param maxNumBoxes The maximum number of bounding boxes of detected
   * objects. There can be multiple objects of the same class, but at different
   * locations. Defaults to 20.
   * @param minScore The minimum score of the returned bounding boxes
   * of detected objects. Value between 0 and 1. Defaults to 0.5.
   */
  async detect(
      img: tf.Tensor3D|ImageData|HTMLImageElement|HTMLCanvasElement|
      HTMLVideoElement,
      maxNumBoxes = 20, minScore = 0.3): Promise<DetectedObject[]> {
    return this.infer(img, maxNumBoxes, minScore);
  }

  /**
   * Dispose the tensors allocated by the model. You should call this when you
   * are done with the model.
   */
  dispose() {
    if (this.model != null) {
      this.model.dispose();
    }
  }
}
