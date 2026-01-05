import * as yolo11 from './model';

const fontSize = 16;
const color = 'aqua';
const lineWidth = 2;

let model: yolo11.ObjectDetection;

export const load = async () => {
  model = await yolo11.load({
    modelUrl: 'https://objectstorageapi.bja.sealos.run/b75r8m7e-models/yolo11n_tfjs/model.json'
  });

  // model = await yolo11.load();
  console.log('Model loaded=', model);
};

export const isReady = () => {
  return !!model;
};

export const dispose = () => {
  model.dispose();
};

export const detect = async (frame: any) => {
  console.log('检测开始，frame尺寸:', frame.width, 'x', frame.height);
  
  // 直接传递原始frame给model.detect，预处理逻辑已移至model.ts内部
  console.log('准备检测，直接使用原始frame');
  
  // 使用更低的阈值，并添加最大检测数量参数
  const detectedObjects = await model.detect(frame, 100, 0.3);
  
  console.log('检测结果:', detectedObjects.length, '个目标，详细:', detectedObjects);
  
  return detectedObjects;
}

export const drawBoxes = (ctx: WechatMiniprogram.CanvasContext, detectedObjects: yolo11.DetectedObject[]) => {
  if (!ctx || !detectedObjects) {
    console.log('drawBoxes: ctx或detectedObjects为空');
    return false;
  }

  console.log('drawBoxes: 检测到', detectedObjects.length, '个目标');
  
  // 降低阈值，显示更多可能的目标
  const minScore = 0.1

  ctx.setFontSize(fontSize);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  detectedObjects.forEach((detectedObject, index) => {
    console.log('目标', index, ':', detectedObject['class'], '分数:', detectedObject.score, '位置:', detectedObject.bbox);
    
    if (detectedObject.score >= minScore) {
      ctx.rect(...(detectedObject.bbox));
      ctx.stroke();

      ctx.setFillStyle(color);
      ctx.fillText(detectedObject['class'] + ':' + detectedObject.score.toFixed(2), detectedObject.bbox[0], detectedObject.bbox[1] - 5);
    }
  });

  ctx.draw();
  return true;
};
