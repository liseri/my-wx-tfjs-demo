const modelWidth = 640.0;
const modelHeight = 640.0;
const modelChannel = 3;

var inferenceStart;
var inferenceEnd;

// COCO数据集的80个类别
const classNameStrs = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush";
const classNames = classNameStrs.split(',');

export class ObjectDetection {

  // 图像显示尺寸结构体 { width: Number, height: Number }
  displaySize;

  // net inference session
  session;

  // is ready
  ready;

  // the predicted class
  mPredClass = "None";

  speedTime = 0.0;

  modelInput = null;

  constructor(displaySize) {
    this.displaySize = {
      width: displaySize.width,
      height: displaySize.height,
    };

    this.modelInput = new Float32Array(modelWidth * modelHeight * modelChannel);
  
    this.ready = false;
  }

  load() {
    return new Promise((resolve, reject) => {

       const modelPath = `${wx.env.USER_DATA_PATH}/yolo11s_roboflow_12.onnx`;

       // 判断之前是否已经下载过onnx模型
        wx.getFileSystemManager().access({
          path: modelPath,
          success: (res) =>
          {
            console.log("file already exist at: " + modelPath)
              this.createInferenceSession(modelPath).then(() =>
              {
                resolve();
              })
          },
          fail: (res) => {
            console.error(res)
            console.log("begin download model");

            const cloudPath = 'https://objectstorageapi.bja.sealos.run/b75r8m7e-models/yolo11n_tfjs/yolo11s_roboflow_12.onnx'
            this.downloadFile(cloudPath, function(r) {
              console.log(`下载进度：${r.progress}%，已下载${r.totalBytesWritten}B，共${r.totalBytesExpectedToWrite}B`)
            }).then(result => {
        
              wx.getFileSystemManager().saveFile({
                tempFilePath:result.tempFilePath,
                filePath: modelPath,
                success: (res) => { // 注册回调函数
                  console.log(res)
                  // const modelPath = res.savedFilePath +'/yolo11s_roboflow_12.onnx'
    
                  const modelPath = res.savedFilePath;
                  console.log("save onnx model at path: " + modelPath)

                  this.createInferenceSession(modelPath).then(() => {
                    resolve();
                  })
                },
                fail(res) {
                  console.error(res)
                  return
                }
              })
        });
          }
        })
    })
  }

  createInferenceSession(modelPath) {
    return new Promise((resolve, reject) => {
      this.session = wx.createInferenceSession({
        model: modelPath,
        /* 0: Lowest  precision e.g., LS16 + A16 + Winograd A16 + approx. math
           1: Lower   precision e.g., LS16 + A16 + Winograd off + approx. math
           2: Modest  precision e.g., LS16 + A32 + Winograd A32 + approx. math
           3: Higher  precision e.g., LS32 + A32 + Winograd A32 + approx. math
           4: Highest precision e.g., LS32 + A32 + Winograd A32 + precise math

           Higher precision always require longer time to run session
        */
        precisionLevel : 0,
        allowNPU : false,     // wheather use NPU for inference, only useful for IOS
        allowQuantize: false, // wheather generate quantize model
      });

      // 监听error事件
      this.session.onError((error) => {
        console.error(error);
        reject(error);
      });
      this.session.onLoad(() => {
        this.ready = true;
        resolve();
      });
    })
  }

  downloadFile(url, onCall = () => {}) {
    return new Promise((resolve, reject) => {
      const task = wx.downloadFile({
        url: url,
        success: res => resolve(res),
        fail: e => {
          const info = e.toString()
          if (info.indexOf('abort') != -1) {
            reject(new Error('【文件下载失败】中断下载'))
          } else {
            reject(new Error('【文件下载失败】网络或其他错误'))
          }
        }
      })
      task.onProgressUpdate((res) => {
        if (onCall(res) == false) {
          task.abort()
        }
      })
    })
  }

  isReady() {
    return this.ready;
  }
  
  predClass() {
    return this.mPredClass;
  }

  // 图像元数据，用于坐标还原
  imageMetadata = {
    scale: 1.0,
    padding: [0, 0],
    originalSize: { width: 0, height: 0 }
  };

  // input is rgba uint8 data
  preProcess(frame, dstInput) {

    return new Promise((resolve, reject) =>
    {
      const origData = new Uint8Array(frame.data);

      // 计算缩放比例，保持宽高比
      const scale = Math.min(modelWidth / frame.width, modelHeight / frame.height);
      
      // 计算新尺寸
      const newWidth = Math.round(frame.width * scale);
      const newHeight = Math.round(frame.height * scale);
      
      // 计算填充
      const padLeft = Math.floor((modelWidth - newWidth) / 2);
      const padTop = Math.floor((modelHeight - newHeight) / 2);
      
      // 保存图像元数据，用于后续坐标还原
      this.imageMetadata = {
        scale: scale,
        padding: [padLeft, padTop],
        originalSize: { width: frame.width, height: frame.height }
      };

      const origHStride = frame.width * 4;
      const origWStride = 4;
    
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];
      const ratio = 1 / 255.0;

      var idx = 0;
      for (var c = 0; c < modelChannel; ++c)
      {
        for (var h = 0; h < modelHeight; ++h)
        {
          for (var w = 0; w < modelWidth; ++w)
          {
            // 计算原始图像坐标
            const origH = Math.round((h - padTop) / scale);
            const origW = Math.round((w - padLeft) / scale);
            
            // 检查是否在有效区域内
            if (origH >= 0 && origH < frame.height && origW >= 0 && origW < frame.width) {
              const origIndex = origH * origHStride + origW * origWStride + c;
              const pixelValue = origData[origIndex] * ratio;
              // 归一化
              var val = (pixelValue - mean[c]) / std[c];
              dstInput[idx] = val;
            } else {
              // 填充区域设置为0
              dstInput[idx] = 0;
            }
            idx++;
          }
        }
      } 

      resolve();
    });

  }
    // 运行推理并获取目标检测结果
  async detect(frame, maxNumBoxes = 20, minScore = 0.3)
  {
    return new Promise((resolve, reject) =>
    {
      this.preProcess(frame, this.modelInput).then(() => {
        const xinput = {
          shape: [1, 3, 640, 640],  // Input data shape in NCHW
          data: this.modelInput.buffer,
          type: 'float32',  // Input data type
        };

        inferenceStart = new Date().getTime()

        this.session.run({
          // Here string "input" Should be the same with the input name in onnx file
          "images": xinput,
        })
        .then((res) => {
          inferenceEnd = new Date().getTime();
   
          this.speedTime = inferenceEnd - inferenceStart
  
          // Here use res.outputname.data, outputname 
          // Should be the same with the output name in onnx file
          let output = new Float32Array(res.output.data);

          // YOLO 模型输出处理
          // 假设输出格式为: [batch, num_boxes, 4 + 1 + num_classes]
          // 其中 4 是边界框坐标 (x, y, w, h), 1 是置信度, 其余是类别概率
          const batchSize = 1;
          const numBoxes = output.length / (4 + 1 + classNames.length);
          const boxSize = 4 + 1 + classNames.length;

          const detections = [];

          // 处理每个检测框
          for (let i = 0; i < numBoxes; i++) {
            const boxOffset = i * boxSize;
            
            // 提取边界框坐标 (x, y, w, h)
            const x = output[boxOffset];
            const y = output[boxOffset + 1];
            const w = output[boxOffset + 2];
            const h = output[boxOffset + 3];
            
            // 提取置信度
            const confidence = output[boxOffset + 4];
            
            // 提取类别概率并找到最高概率的类别
            let maxClassProb = 0;
            let classId = 0;
            
            for (let j = 0; j < classNames.length; j++) {
              const classProb = output[boxOffset + 5 + j];
              if (classProb > maxClassProb) {
                maxClassProb = classProb;
                classId = j;
              }
            }
            
            // 计算最终置信度 (置信度 * 类别概率)
            const finalConfidence = confidence * maxClassProb;
            
            // 应用置信度阈值过滤
            if (finalConfidence >= minScore) {
              // 将 (x, y, w, h) 转换为 (x1, y1, x2, y2)
              const x1 = x - w / 2;
              const y1 = y - h / 2;
              const x2 = x + w / 2;
              const y2 = y + h / 2;
              
              detections.push([x1, y1, x2, y2, finalConfidence, classId]);
            }
          }

          // 应用非极大值抑制 (NMS) 去除重叠框
          const nmsThreshold = 0.45;
          const nmsDetections = this.nms(detections, nmsThreshold);
          
          // 限制检测框数量
          const limitedDetections = nmsDetections.slice(0, maxNumBoxes);
          
          // 坐标还原：将模型输出空间的坐标转换回原始图像空间
          const scaledDetections = limitedDetections.map(box => this.scaleBoxes(box));
          
          // 构建最终检测结果
          const results = scaledDetections.map(box => {
            const [x1, y1, x2, y2, confidence, classId] = box;
            return {
              bbox: [x1, y1, x2 - x1, y2 - y1],  // [x, y, width, height]
              confidence: confidence,
              classId: classId,
              className: classNames[classId]
            };
          });

          resolve(results);
        }).catch(error => {
          console.error(error);
          reject(error);
        });
      })

    });
  }

  // 非极大值抑制 (NMS) 函数
  nms(detections, iouThreshold) {
    // 按置信度降序排序
    detections.sort((a, b) => b[4] - a[4]);
    
    const selectedDetections = [];
    
    while (detections.length > 0) {
      // 选择置信度最高的检测框
      const current = detections.shift();
      selectedDetections.push(current);
      
      // 过滤掉与当前检测框重叠度高的检测框
      detections = detections.filter(box => {
        return this.iou(current, box) < iouThreshold;
      });
    }
    
    return selectedDetections;
  }
  
  // 计算两个边界框的交并比 (IoU)
  iou(box1, box2) {
    const [x1_1, y1_1, x2_1, y2_1] = box1;
    const [x1_2, y1_2, x2_2, y2_2] = box2;
    
    // 计算交集区域
    const x1 = Math.max(x1_1, x1_2);
    const y1 = Math.max(y1_1, y1_2);
    const x2 = Math.min(x2_1, x2_2);
    const y2 = Math.min(y2_1, y2_2);
    
    // 计算交集面积
    const intersectionArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    
    // 计算两个边界框的面积
    const area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    const area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    
    // 计算并集面积
    const unionArea = area1 + area2 - intersectionArea;
    
    // 计算 IoU
    return intersectionArea / unionArea;
  }

  // 坐标还原函数：将模型输出的坐标转换回原始图像空间
  scaleBoxes(box) {
    const { scale, padding, originalSize } = this.imageMetadata;
    const [origWidth, origHeight] = [originalSize.width, originalSize.height];
    const [padLeft, padTop] = padding;
    
    let [x1, y1, x2, y2, confidence, classId] = box;
    
    // 将坐标从模型输出空间转换回原图空间
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

  getTime()
  {
    return this.speedTime;
  }
  dispose() {
    this.session.destroy();
  }
}